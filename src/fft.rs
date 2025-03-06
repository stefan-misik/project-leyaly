use ringbuffer::{AllocRingBuffer, ConstGenericRingBuffer, RingBuffer};
use rustfft::{Fft, FftPlanner, num_complex::Complex, num_traits::Zero};
use std::{ops::Mul, sync::Arc};

pub struct SimpleFft {
    planner: FftPlanner<f32>,
    fft: Arc<dyn Fft<f32>>,
    input_buffer: AllocRingBuffer<Complex<f32>>,
    freq_buffer: Vec<f32>,
}

impl SimpleFft {
    pub fn new(freq_bins: usize) -> SimpleFft {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(1);

        let mut ret = SimpleFft {
            planner,
            fft,
            input_buffer: AllocRingBuffer::new(1),
            freq_buffer: vec![],
        };
        ret.set_freq_bins(freq_bins);

        ret
    }

    pub fn set_freq_bins(&mut self, freq_bins: usize) {
        let sample_rate = freq_bins * 2;

        self.fft = self.planner.plan_fft_forward(sample_rate);
        self.input_buffer = AllocRingBuffer::new(sample_rate);
        self.input_buffer.fill(Complex::zero());
        self.freq_buffer.resize(freq_bins, 0.0);
    }

    /// Processes audio samples for frequency analysis.
    ///
    /// Accumulates samples in an internal buffer and performs FFT processing.
    /// After processing, frequency data can be retrieved with `self.frequencies()`.
    pub fn feed_samples(&mut self, samples: &[f32]) -> Option<()> {
        let complex_samples = samples.iter().map(|s| Complex::<f32>::new(*s, 0.0));
        self.input_buffer.extend(complex_samples);

        let fft_len = self.input_buffer.len();
        let mut buff: Vec<Complex<f32>> = self.input_buffer.iter().copied().collect();

        self.fft.process(buff.as_mut_slice());

        self.freq_buffer = buff
            .iter()
            .map(|s| s.norm().mul(1.0 / fft_len as f32))
            .take(fft_len / 2)
            .collect();

        Some(())
    }

    /// Returns the frequency spectrum data with the specified number of frequency bins.
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
        const SAMPLE_RATE: usize = 40_000;
        const FREQ_BINS: usize = SAMPLE_RATE / 2;
        const TIME_DURATION: usize = 10; // seconds

        const SAMPLES_COUNT: usize = SAMPLE_RATE * TIME_DURATION;

        let mut fft = SimpleFft::new(FREQ_BINS);

        let buffer: Vec<f32> = (0..SAMPLES_COUNT)
            .map(|i| (i as f32) / SAMPLE_RATE as f32)
            .map(|s| 2.0 * PI * s)
            .map(|s| 42.0 + (s * 10.0).sin() + (s * 100.0).sin() / 2.0 + (s * 420.0).sin() / 4.0)
            .collect();

        fft.feed_samples(&buffer[0..40000]);

        let freqs = fft.frequencies();

        assert_eq!(freqs.len(), FREQ_BINS);
        assert_relative_eq!(freqs[0], 42.0, max_relative = 0.05);
        assert_relative_eq!(freqs[10], 0.5, max_relative = 0.05);
        assert_relative_eq!(freqs[100], 0.25, max_relative = 0.05);
        assert_relative_eq!(freqs[420], 0.125, max_relative = 0.05);
    }
}
