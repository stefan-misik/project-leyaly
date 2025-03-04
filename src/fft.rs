use std::{ops::Mul, sync::Arc};

use rustfft::{Fft, FftPlanner, num_complex::Complex};

pub struct SimpleFft {
    fft: Arc<dyn Fft<f32>>,
    input_buffer: Vec<Complex<f32>>,
    freq_buffer: Vec<f32>,
}

impl SimpleFft {
    pub fn new(window_size: usize) -> SimpleFft {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(window_size);
        SimpleFft {
            fft,
            input_buffer: vec![],
            freq_buffer: vec![0.0; window_size / 2],
        }
    }

    pub fn feed_samples(&mut self, samples: &[f32]) {
        let complex_samples = samples.iter().map(|s| Complex::<f32>::new(*s, 0.0));
        self.input_buffer.extend(complex_samples);

        let fft_len = self.fft.len();

        if self.input_buffer.len() < fft_len {
            return;
        }

        let chunks_count = self.input_buffer.len() / fft_len;
        let (samples, leftover) = self.input_buffer.split_at_mut(chunks_count * fft_len);

        let samples_len = samples.len();
        let samples_to_feed = &mut samples[samples_len - fft_len..];

        self.fft.process(samples_to_feed);
        self.freq_buffer = samples_to_feed
            .iter()
            .map(|s| s.norm().mul(1.0 / fft_len as f32))
            .take(fft_len / 2)
            .collect();
        self.input_buffer = leftover.to_vec();
    }

    pub fn frequencies(&self) -> Vec<f32> {
        self.freq_buffer.clone()
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn it_calculates_fft() {
        const SAMPLES_COUNT: usize = 40000;
        const WINDOW_SIZE: usize = 20000;

        let mut fft = SimpleFft::new(WINDOW_SIZE);

        let buffer: Vec<f32> = (0..SAMPLES_COUNT)
            .map(|i| (i as f32) / WINDOW_SIZE as f32)
            .map(|s| 2.0 * PI * s)
            .map(|s| 42.0 + s.sin() + (s * 2.0).sin() / 2.0 + (s * 3.0).sin() / 4.0)
            .collect();

        fft.feed_samples(&buffer);

        let freqs = fft.frequencies();

        assert_eq!(freqs.len(), WINDOW_SIZE / 2);
        assert_relative_eq!(freqs[0], 42.0, max_relative = 0.1);
        assert_relative_eq!(freqs[1], 0.5, max_relative = 0.1);
        assert_relative_eq!(freqs[2], 0.25, max_relative = 0.1);
        assert_relative_eq!(freqs[3], 0.125, max_relative = 0.1);
    }
}
