mod fft;

use std::f32::consts::PI;

use crate::fft::SimpleFft;

fn main() {
    const SAMPLES_COUNT: usize = 1000;
    const WINDOW_SIZE: usize = 16;
    let mut fft = SimpleFft::new(WINDOW_SIZE);

    let buffer: Vec<f32> = (0..SAMPLES_COUNT)
        .map(|i| (i as f32) / WINDOW_SIZE as f32)
        .map(|s| 2.0 * PI * s)
        .map(|s| 42.0 + s.sin() + (s * 2.0).sin() / 2.0 + (s * 3.0).sin() / 4.0)
        .collect();

    println!("input: {:?}\n", buffer);
    fft.feed_samples(&buffer);

    println!("frequencies: {:?}\n", fft.frequencies());
    println!();
}
