use anyhow::{ensure, Context, Result};
use hound::{SampleFormat, WavReader, WavSpec};
use plotters::prelude::*;
use rustfft::{num_complex::Complex, num_traits::Zero, FFTplanner};
use std::{ffi::OsStr, fs::File, path::PathBuf, process::exit, time::Duration};
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
struct Args {
    /// Time step.
    #[structopt(long, parse(try_from_str = humantime::parse_duration), default_value = "1ms")]
    time_step: Duration,

    /// Window size.
    #[structopt(long = "window", parse(try_from_str = humantime::parse_duration), default_value = "30ms")]
    window_duration: Duration,

    /// Minimal frequency, Hz.
    #[structopt(long = "min_freq", default_value = "100")]
    minimal_frequency: usize,

    /// Maximal frequency, Hz.
    #[structopt(long = "max_freq", default_value = "1500")]
    maximal_frequency: usize,

    /// Output heat map in decibels.
    #[structopt(long)]
    decibels: bool,

    /// Input WAV file.
    input: PathBuf,

    /// Output WAV file.
    #[structopt(default_value = "frequencies.png")]
    output: PathBuf,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("{}", e);
        for e in e.chain().skip(1) {
            eprintln!("  -- {}", e);
        }
        exit(1);
    }
}

fn make_even(x: usize) -> usize {
    x + (x % 2 == 1) as usize
}

fn run() -> Result<()> {
    let Args {
        time_step,
        window_duration,
        minimal_frequency,
        maximal_frequency,
        input,
        decibels,
        output,
    } = Args::from_args();

    let syllable = input
        .file_stem()
        .and_then(OsStr::to_str)
        .with_context(|| format!("Unable to extract a file stem from {}", input.display()))?;
    let file = File::open(&input)
        .with_context(|| format!("Unable to open input '{}'", input.display()))?;
    let wav =
        WavReader::new(file).with_context(|| format!("Unable to parse '{}'", input.display()))?;
    let WavSpec {
        channels,
        sample_format,
        sample_rate,
        bits_per_sample,
        ..
    } = wav.spec();
    if channels != 1 {
        eprintln!("Channels will be interleaved");
    }
    ensure!(
        sample_format == SampleFormat::Int,
        "Only WAVE_FORMAT_PCM is supported"
    );
    ensure!(
        bits_per_sample <= 16,
        "Too much bits per sample: {}",
        bits_per_sample
    );

    let samples = wav.into_samples::<i16>();

    let total_samples = samples.len();
    let total_time = total_samples as f32 / sample_rate as f32;
    println!("Time = {}s, sample rate = {} Hz", total_time, sample_rate);
    let samples = samples
        .enumerate()
        .map(|(id, sample)| sample.map(|sample| (id as f32 / sample_rate as f32, sample as f32)))
        .collect::<Result<Vec<_>, _>>()
        .with_context(|| format!("Unable to read a sample from '{}'", input.display()))?;

    let time_step_points = (time_step.as_nanos() as f32 / 1e9 * sample_rate as f32) as usize;
    let time_step_sec = time_step_points as f32 / sample_rate as f32;

    let windows_points =
        make_even((window_duration.as_nanos() as f32 / 1e9 * sample_rate as f32) as usize);
    let windows_sec = windows_points as f32 / sample_rate as f32;
    let frequency_width = 0.5f32 / windows_sec;
    println!(
        "Scanning window: {} points ({:.2} ms, freq width = {}); time step: {} points ({:.2} µs)",
        windows_points,
        windows_sec * 1000.,
        frequency_width,
        time_step_points,
        (time_step_points as f32 / sample_rate as f32) * 1e6
    );

    let mut heat_map = vec![];

    let mut min_power = 1e9_f32;
    let mut max_power = 0f32;

    for i in 0.. {
        let time_point = time_step_points * i;
        if time_point > total_samples - windows_points {
            break;
        }
        let mut input: Vec<Complex<f32>> = samples
            .iter()
            .skip(time_point)
            .take(windows_points)
            .map(|&(_t, value)| Complex::new(value, 0.))
            .collect();
        assert_eq!(input.len(), windows_points);
        let mut output: Vec<Complex<f32>> = vec![Zero::zero(); windows_points];

        let mut planner = FFTplanner::new(false);
        let fft = planner.plan_fft(windows_points);
        fft.process(&mut input, &mut output);
        let power_per_frequency = output
            .into_iter()
            .enumerate()
            .take(windows_points / 2)
            .filter_map(|(id, c)| {
                let frequency = id as f32 / windows_sec;
                if frequency > maximal_frequency as f32 || frequency < minimal_frequency as f32 {
                    None
                } else {
                    let power = c.norm_sqr() / windows_points as f32;
                    max_power = max_power.max(power);
                    min_power = min_power.min(power);
                    Some(power)
                }
            })
            .collect::<Vec<_>>();
        heat_map.push(power_per_frequency);
    }

    let root = BitMapBackend::new(&output, (1280, 2 * 480)).into_drawing_area();
    root.fill(&WHITE).context("Fill areas")?;
    let root = root.margin(10, 10, 10, 10);

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("{} frequency power heat map", syllable),
            ("sans-serif", 40).into_font(),
        )
        .x_label_area_size(20)
        .y_label_area_size(40)
        .build_cartesian_2d(
            0f32..total_time,
            (minimal_frequency as f32..maximal_frequency as f32 + 1.)
                .log_scale()
                .with_key_points(
                    (minimal_frequency..=maximal_frequency)
                        .filter_map(|val| {
                            if val < 500 && val % 100 == 0
                                || val < 2000 && val % 200 == 0
                                || val % 500 == 0
                            {
                                Some(val as f32)
                            } else {
                                None
                            }
                        })
                        .collect(),
                ),
        )
        .context("Chart builder")?;

    chart
        .configure_mesh()
        .x_labels(5)
        .x_desc("Time (s)")
        .y_desc("Frequency (Hz)")
        .y_labels(5)
        .disable_mesh()
        .draw()
        .context("Mesh")?;

    let plotting_area = chart.plotting_area();

    let min_power = min_power.max(1f32);

    let fn1;
    let fn2;
    let normalize_output: &dyn Fn(f32) -> f32 = if decibels {
        let min_db = (min_power / max_power).log10() * 20f32;
        fn1 = move |power: f32| ((power / max_power).log10() * 20f32 - min_db) / min_db.abs();
        &fn1
    } else {
        fn2 = |power: f32| power / max_power;
        &fn2
    };

    for (time, data_per_time) in heat_map.into_iter().enumerate() {
        let time = (time * time_step_points) as f32 / sample_rate as f32;

        let mut local_max_frequency = 0f32;
        let mut local_max_power = 0f32;

        for (id, power) in data_per_time.into_iter().enumerate() {
            let frequency = id as f32 / windows_sec;
            let normalized_power = normalize_output(power);

            if normalized_power >= local_max_power {
                local_max_power = normalized_power;
                local_max_frequency = frequency;
            }

            let color = HSLColor(184. / 255., 1.0, 1.0 - normalized_power as f64 * 0.5).filled();

            let rect = Rectangle::new(
                [
                    (time, frequency + frequency_width),
                    (time + time_step_sec, frequency - frequency_width),
                ],
                color,
            );
            plotting_area.draw(&rect).context("Rectangle")?;
        }
        // Mark the maximal point at this time frame.
        plotting_area
            .draw(&Rectangle::new(
                [
                    (time, local_max_frequency + frequency_width),
                    (time + time_step_sec, local_max_frequency - frequency_width),
                ],
                HSLColor(184. / 255., 1.0, 0.5).filled(),
            ))
            .context("Max rectangle")?;
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn check_make_even() {
        assert_eq!(make_even(0), 0);
        assert_eq!(make_even(1), 2);
        assert_eq!(make_even(2), 2);
    }
}
