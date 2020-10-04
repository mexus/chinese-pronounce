use anyhow::{ensure, Context, Result};
use hound::{SampleFormat, WavReader, WavSpec};
use plotters::prelude::*;
use rustfft::{num_complex::Complex, num_traits::Zero, FFTplanner};
use std::{ffi::OsStr, fs::File, path::PathBuf, process::exit};
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
struct Args {
    /// Input WAV file.
    input: PathBuf,
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

fn run() -> Result<()> {
    let Args { input } = Args::from_args();

    let stem = input
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

    let time_step_points = {
        let time_step_sec = 0.001f32; // 1ms
        time_step_sec * sample_rate as f32
    } as usize;
    let time_step_sec = time_step_points as f32 / sample_rate as f32;

    let windows_points = 2000usize;
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

    let min_frequency = 100;
    let max_frequency = 2500.min(sample_rate / 2);
    let mut max_value = 0f32;

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
        let values_per_frequency = output
            .into_iter()
            .enumerate()
            .take(windows_points / 2)
            .filter_map(|(id, c)| {
                let frequency = id as f32 / windows_sec;
                if frequency > max_frequency as f32 || frequency < min_frequency as f32 {
                    None
                } else {
                    let value = (c.norm_sqr() / windows_points as f32).sqrt();
                    max_value = max_value.max(value);
                    Some(value)
                }
            })
            .collect::<Vec<_>>();
        heat_map.push(values_per_frequency);
    }

    let root = BitMapBackend::new("frequencies.png", (1280, 2 * 480)).into_drawing_area();
    root.fill(&WHITE).context("Fill areas")?;
    let root = root.margin(10, 10, 10, 10);

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("{} frequencies heat map", stem),
            ("sans-serif", 40).into_font(),
        )
        .x_label_area_size(20)
        .y_label_area_size(40)
        .build_cartesian_2d(
            0f32..total_time,
            (min_frequency as f32..max_frequency as f32 + 1.)
                .log_scale()
                .with_key_points(
                    (min_frequency..=max_frequency)
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

    for (time, data_per_time) in heat_map.into_iter().enumerate() {
        let time = (time * time_step_points) as f32 / sample_rate as f32;

        let mut local_max_frequency = 0f32;
        let mut local_max_value = 0f32;
        for (id, value) in data_per_time.into_iter().enumerate() {
            let frequency = id as f32 / windows_sec;
            // println!("{:?}", frequency);

            let value = value / max_value;

            if value >= local_max_value {
                local_max_value = value;
                local_max_frequency = frequency;
            }

            let color = HSLColor(184. / 255., 1.0, 1.0 - value as f64 * 0.5).filled();

            let rect = Rectangle::new(
                [
                    (time, frequency + frequency_width / 2.),
                    (time + time_step_sec, frequency - frequency_width / 2.),
                ],
                color,
            );
            plotting_area.draw(&rect).context("Rectangle")?;
            // plotting_area.draw_pixel((time, frequency), &color)?
        }
        // Mark the maximal point at this time frame.
        plotting_area
            .draw(&Rectangle::new(
                [
                    (time, local_max_frequency + frequency_width / 2.),
                    (
                        time + time_step_sec,
                        local_max_frequency - frequency_width / 2.,
                    ),
                ],
                HSLColor(184. / 255., 1.0, 0.5).filled(),
            ))
            .context("Max rectangle")?;
        // plotting_area.draw_pixel((time + time_step_sec, local_max_frequency), &RED)?
    }

    Ok(())
}
