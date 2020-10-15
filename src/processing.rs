//! Processing helpers.

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
            window.as_ref().len() % 2 == 1,
            "Window size must NOT be divisible by two"
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
    ///
    /// Single window at time `t`:
    ///         t
    ///         -
    ///        / \
    ///       /   \
    ///      /     \
    /// t-w/2  ...  t+w/2
    ///
    /// time from 0 to 9 (length = 10), windows of size 7 (w/2 = 3):
    /// 1. `t` from -3 to 3 (top at `t` = 0)
    /// 2. `t` from 0 to 6  (top at `t` = 3)
    /// 3. `t` from 3 to 9  (top at `t` = 6)
    /// 4. `t` from 6 to 12 (top at `t` = 9)
    ///
    /// t =     0  3  6  9
    ///         -  -  -  -   
    ///        / \/ \/ \/ \  
    ///       /  /\ /\ /\  \
    ///      /  /  X  X  \  \
    /// t = -3  0  3  6  9  12
    ///
    /// Base points count = length / (w/2) + 1 = 10 / 3 + 1 = 3 + 1 = 4
    pub fn process(&self) -> ProcessingResult {
        let window = self.window.as_ref();
        let window_size = window.len();

        let half_width = window_size / 2;

        let samples = self.samples.as_ref();
        let samples_count = samples.len();

        let mut temp_input: Vec<Complex<f32>> = vec![Zero::zero(); window_size];
        let mut temp_output: Vec<Complex<f32>> = vec![Zero::zero(); window_size];

        let mut heat_map = Array2::zeros((samples_count / half_width + 1, samples_count / 2));

        let mut planner = FFTplanner::new(false);
        let fft = planner.plan_fft(window_size);

        let mut minimum_value = f32::MAX;
        let mut maximum_value = f32::MIN;

        for (index, heat_row) in heat_map.genrows_mut().into_iter().enumerate() {
            let frame_begin = index * half_width - half_width;
            let frame_end = index * half_width + half_width;

            dbg!(frame_begin..=frame_end)
                .zip(&mut temp_input)
                .zip(window)
                .for_each(|((t, input), &window_coefficient)| {
                    *input = Complex::new(
                        samples
                            .get(t)
                            .map(|&value| value as f32 * window_coefficient)
                            .unwrap_or(0.),
                        0.,
                    );
                });

            fft.process(&mut temp_input, &mut temp_output);

            let heat_row: &mut [f32] = heat_row.into_slice().expect("Contiguous array");
            heat_row
                .iter_mut()
                .zip(&temp_output)
                .for_each(|(output, temp_output)| {
                    let value = temp_output.norm_sqr() / window_size as f32;
                    minimum_value = minimum_value.min(value);
                    maximum_value = maximum_value.max(value);
                    *output = value;
                });
        }

        ProcessingResult {
            heat_map,
            minimum_value,
            maximum_value,
        }
    }
}

/// Processing result.
pub struct ProcessingResult {
    pub heat_map: Array2<f32>,
    pub minimum_value: f32,
    pub maximum_value: f32,
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
mod hann_window_test {
    use super::HannWindow;

    #[test]
    fn check_zero() {
        for length in 2..1024 {
            let window = HannWindow::new(length);
            assert!(window.coefficient(0) < std::f32::MIN_POSITIVE);
            assert!(window.coefficient(length - 1) < std::f32::MIN_POSITIVE);
        }
    }

    #[test]
    fn check_symmetry() {
        for length in 2..1024 {
            if length % 2 == 0 {
                // If length % 2 == 0, there is no central point.
                continue;
            }
            let window = HannWindow::new(length);
            for i in 0..length / 2 {
                let left = window.coefficient(length / 2 - i);
                let right = window.coefficient(length / 2 + i);
                let diff = left - right;
                assert!(
                    diff.abs() < std::f32::MIN_POSITIVE,
                    "f({}) - f({}) = {}",
                    length / 2 - i,
                    length / 2 + i,
                    diff
                );
            }
        }
    }

    #[test]
    fn check_peak() {
        for length in 2..1024 {
            if length % 2 == 0 {
                // If length % 2 == 0, there is no central point.
                continue;
            }
            let window = HannWindow::new(length);

            // Peak must be reached at the central point.
            let expected = window.coefficient(length / 2);

            // Test that all the other points contain lesser values.
            let less = (0..length)
                .filter(|&idx| idx != length / 2)
                .all(|idx| window.coefficient(idx) < expected);
            assert!(less, "length = {}", length);
        }
    }
}
