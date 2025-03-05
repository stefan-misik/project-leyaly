use rustfft::{Fft, FftPlanner, num_complex::Complex};
use std::{ops::Mul, sync::Arc};

pub struct SimpleFft {
    fft: Arc<dyn Fft<f32>>,
    input_buffer: Vec<Complex<f32>>,
    freq_buffer: Vec<f32>,
}

impl SimpleFft {
    pub fn new(sample_rate: usize) -> SimpleFft {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(sample_rate);
        SimpleFft {
            fft,
            input_buffer: vec![],
            freq_buffer: vec![0.0; sample_rate / 2],
        }
    }

    /// Processes audio samples for frequency analysis.
    ///
    /// Accumulates samples in an internal buffer and performs FFT when enough samples
    /// are available. After processing, frequency data can be retrieved with `self.frequencies()`.
    ///
    /// Returns:
    /// - `Some(())` if FFT processing was performed (enough samples were available)
    /// - `None` if not enough samples were available to perform FFT
    pub fn feed_samples(&mut self, samples: &[f32]) -> Option<()> {
        let complex_samples = samples.iter().map(|s| Complex::<f32>::new(*s, 0.0));
        self.input_buffer.extend(complex_samples);

        let fft_len = self.fft.len();

        if self.input_buffer.len() < fft_len {
            return None;
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

        Some(())
    }

    /// Returns the frequency spectrum data with the specified number of frequency bins.
    pub fn frequencies(&self, freq_bins: usize) -> Vec<f32> {
        if freq_bins == 0 {
            return Vec::new();
        }

        if freq_bins == self.freq_buffer.len() {
            return self.freq_buffer.clone();
        }

        let mut result = Vec::with_capacity(freq_bins);
        let src_len = self.freq_buffer.len();

        let scale_factor = (src_len as f32) / (freq_bins as f32);

        // nearest-neighbor scaling
        for i in 0..freq_bins {
            let src_idx = (i as f32 * scale_factor).round() as usize;
            let src_idx = src_idx.min(src_len - 1);
            result.push(self.freq_buffer[src_idx]);
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn it_calculates_fft() {
        const SAMPLE_RATE: usize = 40_000;
        const TIME_DURATION: usize = 10; // seconds

        const SAMPLES_COUNT: usize = SAMPLE_RATE * TIME_DURATION;

        let mut fft = SimpleFft::new(SAMPLE_RATE);

        let buffer: Vec<f32> = (0..SAMPLES_COUNT)
            .map(|i| (i as f32) / SAMPLE_RATE as f32)
            .map(|s| 2.0 * PI * s)
            .map(|s| 42.0 + (s * 10.0).sin() + (s * 100.0).sin() / 2.0 + (s * 420.0).sin() / 4.0)
            .collect();

        fft.feed_samples(&buffer[0..40000]);
        fft.feed_samples(&buffer);
        fft.feed_samples(&buffer);

        let freqs = fft.frequencies(SAMPLE_RATE / 2);

        assert_eq!(freqs.len(), SAMPLE_RATE / 2);
        assert_relative_eq!(freqs[0], 42.0, max_relative = 0.05);
        assert_relative_eq!(freqs[10], 0.5, max_relative = 0.05);
        assert_relative_eq!(freqs[100], 0.25, max_relative = 0.05);
        assert_relative_eq!(freqs[420], 0.125, max_relative = 0.05);
    }

    #[ignore]
    #[test]
    fn it_scales_frequency_bins() {
        const SAMPLE_RATE: usize = 40_000;
        const TIME_DURATION: usize = 10; // seconds

        const SAMPLES_COUNT: usize = SAMPLE_RATE * TIME_DURATION;

        let mut fft = SimpleFft::new(SAMPLE_RATE);

        let buffer: Vec<f32> = (0..SAMPLES_COUNT)
            .map(|i| (i as f32) / SAMPLE_RATE as f32)
            .map(|s| 2.0 * PI * s)
            .map(|s| 42.0 + (s * 100.0).sin() + (s * 500.0).sin() / 2.0 + (s * 1000.0).sin() / 4.0)
            .collect();

        fft.feed_samples(&buffer);

        let max_freq = SAMPLE_RATE / 2;
        let scale_factor = 100;
        let freq_bins = max_freq / scale_factor;

        let freqs = fft.frequencies(freq_bins);

        assert_eq!(freqs.len(), freq_bins);
        assert_relative_eq!(freqs[0], 42.0, max_relative = 0.05);
        assert_relative_eq!(freqs[1], 0.5, max_relative = 0.05);
        assert_relative_eq!(freqs[5], 0.25, max_relative = 0.05);
        assert_relative_eq!(freqs[10], 0.125, max_relative = 0.05);
    }

    #[ignore]
    #[test]
    fn it_accepts_smaller_chunks_than_sample_rate() {
        const SAMPLE_RATE: usize = 40_000;
        const TIME_DURATION: usize = 10; // seconds

        const SAMPLES_COUNT: usize = SAMPLE_RATE * TIME_DURATION;

        let mut fft = SimpleFft::new(SAMPLE_RATE);

        let buffer: Vec<f32> = (0..SAMPLES_COUNT)
            .map(|i| (i as f32) / SAMPLE_RATE as f32)
            .map(|s| 2.0 * PI * s)
            .map(|s| 42.0 + s.sin() + (s * 100.0).sin() / 2.0 + (s * 420.0).sin() / 4.0)
            .collect();

        let chunk_len = SAMPLE_RATE / 4;
        fft.feed_samples(&buffer[0..chunk_len]);

        let freqs = fft.frequencies(SAMPLE_RATE / 2);

        assert_eq!(freqs.len(), SAMPLE_RATE / 2);
        assert_relative_eq!(freqs[0], 0.0, max_relative = 0.05);
        assert_relative_eq!(freqs[1], 0.0, max_relative = 0.05);
        assert_relative_eq!(freqs[100], 0.0, max_relative = 0.05);
        assert_relative_eq!(freqs[420], 0.0, max_relative = 0.05);

        fft.feed_samples(&buffer[chunk_len..(chunk_len * 2)]);

        let freqs = fft.frequencies(SAMPLE_RATE / 2);

        assert_eq!(freqs.len(), SAMPLE_RATE / 2);
        assert_relative_eq!(freqs[0], 0.0, max_relative = 0.05);
        assert_relative_eq!(freqs[1], 0.0, max_relative = 0.05);
        assert_relative_eq!(freqs[100], 0.0, max_relative = 0.05);
        assert_relative_eq!(freqs[420], 0.0, max_relative = 0.05);

        fft.feed_samples(&buffer[(chunk_len * 2)..(chunk_len * 3)]);

        let freqs = fft.frequencies(SAMPLE_RATE / 2);

        assert_eq!(freqs.len(), SAMPLE_RATE / 2);
        assert_relative_eq!(freqs[0], 0.0, max_relative = 0.05);
        assert_relative_eq!(freqs[1], 0.0, max_relative = 0.05);
        assert_relative_eq!(freqs[100], 0.0, max_relative = 0.05);
        assert_relative_eq!(freqs[420], 0.0, max_relative = 0.05);

        fft.feed_samples(&buffer[(chunk_len * 3)..(chunk_len * 4)]);

        let freqs = fft.frequencies(SAMPLE_RATE / 2);

        assert_eq!(freqs.len(), SAMPLE_RATE / 2);
        assert_relative_eq!(freqs[0], 42.0, max_relative = 0.05);
        assert_relative_eq!(freqs[1], 0.5, max_relative = 0.05);
        assert_relative_eq!(freqs[100], 0.25, max_relative = 0.05);
        assert_relative_eq!(freqs[420], 0.125, max_relative = 0.05);
    }

    #[ignore]
    #[test]
    fn it_accepts_bigger_chunks_than_sample_rate() {
        const SAMPLE_RATE: usize = 40_000;
        const TIME_DURATION: usize = 10; // seconds

        const SAMPLES_COUNT: usize = SAMPLE_RATE * TIME_DURATION;

        let mut fft = SimpleFft::new(SAMPLE_RATE);

        let buffer: Vec<f32> = (0..SAMPLES_COUNT)
            .map(|i| (i as f32) / SAMPLE_RATE as f32)
            .map(|s| 2.0 * PI * s)
            .map(|s| 42.0 + s.sin() + (s * 100.0).sin() / 2.0 + (s * 420.0).sin() / 4.0)
            .collect();

        let chunk_len = SAMPLE_RATE + 42;
        fft.feed_samples(&buffer[0..chunk_len]);

        let freqs = fft.frequencies(SAMPLE_RATE / 2);

        assert_eq!(freqs.len(), SAMPLE_RATE / 2);
        assert_relative_eq!(freqs[0], 42.0, max_relative = 0.05);
        assert_relative_eq!(freqs[1], 0.5, max_relative = 0.05);
        assert_relative_eq!(freqs[100], 0.25, max_relative = 0.05);
        assert_relative_eq!(freqs[420], 0.125, max_relative = 0.05);
    }
}
