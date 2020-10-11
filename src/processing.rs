//! Processing helpers.

use itertools::Itertools;
use ndarray::Array2;
use rustfft::{num_complex::Complex, num_traits::Zero, FFTplanner};
use std::f32::consts::PI;

/// Sound processor configuration.
#[derive(Debug, Default, Copy, Clone)]
pub struct ProcessorBuilder<Window, Samples> {
    window: Window,
    samples: Samples,
}

impl ProcessorBuilder<(), ()> {
    /// Creates a new configuration.
    pub fn new() -> Self {
        Self::default()
    }
}

impl<AnyWindow, AnySamples> ProcessorBuilder<AnyWindow, AnySamples> {
    /// Sets audio samples.
    pub fn samples<Samples>(self, samples: Samples) -> ProcessorBuilder<AnyWindow, Samples>
    where
        Samples: AsRef<[i16]>,
    {
        assert!(
            samples.as_ref().len() < 2,
            "At least two samples are required"
        );
        assert!(
            samples.as_ref().len() % 2 == 0,
            "Samples count must be divisible by two"
        );
        ProcessorBuilder {
            samples,
            window: self.window,
        }
    }

    /// Sets window.
    pub fn window<Window>(self, window: Window) -> ProcessorBuilder<Window, AnySamples>
    where
        Window: AsRef<[f32]>,
    {
        assert!(!window.as_ref().is_empty(), "Window must be non-empty");
        assert!(
            window.as_ref().len() % 2 == 0,
            "Window size must be divisible by two"
        );
        ProcessorBuilder {
            window,
            samples: self.samples,
        }
    }

    /// Sets a Hann window.
    pub fn hann_window(self, length: usize) -> ProcessorBuilder<Vec<f32>, AnySamples> {
        self.window(HannWindow::new(length).into_vec())
    }
}

impl<Window, Samples> ProcessorBuilder<Window, Samples>
where
    Samples: AsRef<[i16]>,
    Window: AsRef<[f32]>,
{
    /// Processes the given audio samples using a short-time discrete
    /// fast-fourier transform.
    pub fn process(&self) -> Array2<f32> {
        let window = self.window.as_ref();
        let window_size = window.len();

        let samples = self.samples.as_ref();
        let samples_count = samples.len();

        let mut temp_input: Vec<Complex<f32>> = vec![Zero::zero(); window_size];
        let mut temp_output: Vec<Complex<f32>> = vec![Zero::zero(); window_size];

        let mut heat_map = Array2::zeros((samples_count / window_size, samples_count / 2));

        let mut planner = FFTplanner::new(false);
        let fft = planner.plan_fft(window_size);

        for (samples, heat_row) in (0..)
            .step_by(window_size / 2)
            .map(|time_shift| {
                let samples = samples.get(time_shift..)?;
                samples.get(..window_size)
            })
            .while_some()
            .zip(heat_map.genrows_mut())
        {
            temp_input.iter_mut().zip(samples).zip(window).for_each(
                |((input, &sample), &window_coefficient)| {
                    *input = Complex::new(sample as f32 * window_coefficient, 0.)
                },
            );
            fft.process(&mut temp_input, &mut temp_output);

            let heat_row: &mut [f32] = heat_row.into_slice().expect("Contiguous array");
            debug_assert_eq!(heat_row.len(), samples_count / 2);

            heat_row
                .iter_mut()
                .zip(&temp_output)
                .for_each(|(output, temp_output)| {
                    *output = temp_output.norm_sqr() / window_size as f32;
                });
        }

        heat_map
    }
}

/// Hann window.
pub struct HannWindow {
    length: usize,
}

impl HannWindow {
    /// Creates a Hann window with a given length.
    pub const fn new(length: usize) -> Self {
        Self { length }
    }

    fn coefficient(&self, n: usize) -> f32 {
        debug_assert!(n < self.length);
        let n = n as f32 - (self.length as f32 - 1.) / 2.;
        0.5 * (1.0 + (2.0 * PI * n as f32 / (self.length as f32 - 1.)).cos())
    }

    /// Converts the window into a vector.
    pub fn into_vec(self) -> Vec<f32> {
        (0..self.length).map(|n| self.coefficient(n)).collect()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn check_zero() {
        for length in 2..1024 {
            let window = HannWindow::new(length);
            assert!(window.coefficient(0) < std::f32::MIN_POSITIVE);
            assert!(window.coefficient(length - 1) < std::f32::MIN_POSITIVE);
        }
    }
}
