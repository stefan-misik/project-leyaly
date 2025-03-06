mod fft;

use crate::fft::SimpleFft;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind};
use rand::Rng;
use ratatui::{
    DefaultTerminal, Frame,
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
    widgets::{Block, Sparkline, Widget},
};
use std::{
    f32::consts::PI,
    io,
    ops::Deref,
    time::{Duration, Instant},
};

const MAX_FREQ: usize = 1000;
const SAMPLE_RATE: usize = (MAX_FREQ + 1) * 2;
const SAMPLE_COUNT: usize = SAMPLE_RATE;

#[derive(Debug)]
pub struct App {
    counter: i8,
    exit: bool,
    freq_bins: Vec<f32>,
    last_update: Option<Instant>,
    update_interval: Duration,
    dominant_frequencies: Vec<f32>,
}

impl Default for App {
    fn default() -> Self {
        Self {
            counter: 0,
            exit: false,
            freq_bins: Vec::new(),
            last_update: None,
            update_interval: Duration::from_millis(100),
            dominant_frequencies: vec![],
        }
    }
}

impl App {
    pub fn run(&mut self, terminal: &mut DefaultTerminal, fft: &mut SimpleFft) -> io::Result<()> {
        self.randomize_frequencies();
        self.last_update = Some(Instant::now());

        while !self.exit {
            let now = Instant::now();

            // Check if it's time to generate new data
            if let Some(last_update) = self.last_update {
                if now.duration_since(last_update) >= self.update_interval {
                    self.update_frequencies();
                    self.generate_and_feed_samples(fft);
                    self.last_update = Some(now);

                    terminal.draw(|frame| self.draw(frame, fft))?;
                }
            }

            // Non-blocking event handling with a small timeout
            if crossterm::event::poll(Duration::from_millis(10))? {
                self.handle_events()?;
            }
        }
        Ok(())
    }

    fn update_frequencies(&mut self) {
        let mut rng = rand::rng();

        // Randomly adjust the dominant frequencies
        for freq in &mut self.dominant_frequencies {
            // Add some random variation to each frequency
            let variation = rng.random_range(-5.0f32..5.0f32);
            *freq += variation;

            // Keep frequencies in a reasonable range
            *freq = freq.max(20.0).min(MAX_FREQ as f32);
        }

        self.dominant_frequencies
            .sort_by(|a, b| a.partial_cmp(b).unwrap());
    }

    fn generate_and_feed_samples(&mut self, fft: &mut SimpleFft) {
        let mut rng = rand::rng();

        // Generate a buffer with the current dominant frequencies
        let buffer: Vec<f32> = (0..SAMPLE_COUNT)
            .map(|i| (i as f32) / SAMPLE_RATE as f32)
            .map(|s| 2.0 * PI * s)
            .map(|s| {
                let mut signal = 0.0;

                // Add each dominant frequency with random amplitude
                for (i, &freq) in self.dominant_frequencies.iter().enumerate() {
                    let amplitude = (self.dominant_frequencies.len() - i) as f32 * 2.0;
                    signal += (s * freq).sin() * amplitude;
                }

                // Add some noise
                signal += rng.random_range(-0.2..0.2) * 10.0;

                signal
            })
            .collect();

        fft.feed_samples(&buffer);
    }

    fn draw(&mut self, frame: &mut Frame, fft: &SimpleFft) {
        self.freq_bins = fft.frequencies(frame.area().width as usize);

        frame.render_widget(self.deref(), frame.area());
    }

    fn handle_events(&mut self) -> io::Result<()> {
        match event::read()? {
            Event::Key(key_event) if key_event.kind == KeyEventKind::Press => {
                self.handle_key_event(key_event)
            }
            _ => {}
        };
        Ok(())
    }

    fn handle_key_event(&mut self, key_event: KeyEvent) {
        match key_event.code {
            KeyCode::Char('q') => self.exit(),
            KeyCode::Left => self.decrement_counter(),
            KeyCode::Right => self.increment_counter(),
            KeyCode::Up => self.increase_update_rate(),
            KeyCode::Down => self.decrease_update_rate(),
            KeyCode::Char(' ') => self.randomize_frequencies(),
            _ => {}
        }
    }

    fn increase_update_rate(&mut self) {
        let new_interval = self.update_interval.as_millis().saturating_sub(10);
        self.update_interval = Duration::from_millis(new_interval.max(10) as u64);
    }

    fn decrease_update_rate(&mut self) {
        let new_interval = self.update_interval.as_millis().saturating_add(10);
        self.update_interval = Duration::from_millis(new_interval.min(500) as u64);
    }

    fn randomize_frequencies(&mut self) {
        let mut rng = rand::rng();
        let count = 5;

        self.dominant_frequencies.clear();
        for _ in 0..count {
            self.dominant_frequencies
                .push(rng.random_range(20.0..(MAX_FREQ as f32)));
        }
    }

    fn exit(&mut self) {
        self.exit = true;
    }

    fn increment_counter(&mut self) {
        self.counter = self.counter.wrapping_add(1);
    }

    fn decrement_counter(&mut self) {
        self.counter = self.counter.wrapping_sub(1);
    }
}

impl Widget for &App {
    fn render(self, area: Rect, buf: &mut Buffer) {
        // let min_freq = self.freq_bins.iter().fold(1.0, |a, &b| -> f32 { a.min(b) });

        let freqs: Vec<u64> = self
            .freq_bins
            .iter()
            .map(|s| (s / 0.001).floor() as u64)
            .collect();

        let sparkline = Sparkline::default()
            .block(Block::new().title(format!("{:?}", self.dominant_frequencies)))
            .data(freqs)
            .style(Style::default().fg(Color::LightRed));

        sparkline.render(area, buf);
    }
}

fn main() -> io::Result<()> {
    let mut fft = SimpleFft::new(SAMPLE_RATE);

    let mut terminal = ratatui::init();
    let app_result = App::default().run(&mut terminal, &mut fft);
    ratatui::restore();

    app_result
}
