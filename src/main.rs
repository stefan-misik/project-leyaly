mod fft;

use std::f32::consts::PI;

use crate::fft::SimpleFft;

fn main() {
    const MAX_FREQ: usize = 1000;
    const SAMPLE_RATE: usize = MAX_FREQ * 2;
    const SAMPLE_COUNT: usize = 10 * SAMPLE_RATE;

    let mut fft = SimpleFft::new(SAMPLE_RATE);

    let buffer: Vec<f32> = (0..SAMPLE_COUNT)
        .map(|i| (i as f32) / SAMPLE_RATE as f32)
        .map(|s| 2.0 * PI * s)
        .map(|s| 42.0 + s.sin() + (s * 100.0).sin() / 2.0 + (s * 300.0).sin() / 4.0)
        .collect();

    fft.feed_samples(&buffer);

    println!("frequencies: {:?}\n", fft.frequencies(MAX_FREQ / 100));
    println!();
}
