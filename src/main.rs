mod fft;

use crate::fft::SimpleFft;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind};
use ratatui::{
    DefaultTerminal, Frame,
    buffer::Buffer,
    layout::{Constraint, Layout, Rect},
    style::{Color, Style},
    widgets::{Block, Sparkline, Widget},
};
use std::{
    io,
    ops::Deref,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

pub struct App {
    counter: i8,
    exit: bool,
    freq_bins: Vec<f32>,
    last_update: Option<Instant>,
    update_interval: Duration,
    fft: Arc<Mutex<SimpleFft>>,
}

impl App {
    pub fn new(fft: Arc<Mutex<SimpleFft>>) -> Self {
        App {
            counter: Default::default(),
            exit: Default::default(),
            freq_bins: Default::default(),
            last_update: Default::default(),
            update_interval: Default::default(),
            fft,
        }
    }

    pub fn run(&mut self, terminal: &mut DefaultTerminal) -> io::Result<()> {
        self.last_update = Some(Instant::now());

        if let Ok(mut fft) = self.fft.lock() {
            fft.set_freq_bins(terminal.size().map(|s| s.width).unwrap_or(100) as usize);
        }

        while !self.exit {
            let now = Instant::now();

            // Check if it's time to generate new data
            if let Some(last_update) = self.last_update {
                if now.duration_since(last_update) >= self.update_interval {
                    self.last_update = Some(now);

                    terminal.draw(|frame| self.draw(frame))?;
                }
            }

            // Non-blocking event handling with a small timeout
            if crossterm::event::poll(Duration::from_millis(10))? {
                self.handle_events()?;
            }
        }
        Ok(())
    }

    fn draw(&mut self, frame: &mut Frame) {
        if let Ok(fft) = self.fft.lock() {
            const SKIP_BINS: usize = 1;

            let new_freq_bins: Vec<f32> =
                fft.frequencies().iter().copied().skip(SKIP_BINS).collect();
            self.freq_bins.resize(new_freq_bins.len(), 0.0);

            const ECHO_MULT: f32 = 1.0 / 1.25;
            self.freq_bins = self
                .freq_bins
                .iter()
                .zip(new_freq_bins)
                .map(|(a, b)| (a + b) * ECHO_MULT)
                .collect();
        }

        frame.render_widget(self.deref(), frame.area());
    }

    fn handle_events(&mut self) -> io::Result<()> {
        match event::read()? {
            Event::Key(key_event) if key_event.kind == KeyEventKind::Press => {
                self.handle_key_event(key_event)
            }
            Event::Resize(width, _) => {
                // TODO: this does nothing
                if let Ok(mut fft) = self.fft.lock() {
                    fft.set_freq_bins(width as usize);
                }
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
        // TODO: logarithmic scale
        let min_amplitude = self.freq_bins.iter().fold(1.0, |a, &b| -> f32 { a.min(b) });
        // let min_amplitude: f32 = 0.00000001;

        let freqs: Vec<u64> = self
            .freq_bins
            .iter()
            .map(|s| (s / min_amplitude).floor() as u64)
            .collect();

        let chunks = Layout::vertical(Constraint::from_percentages([50, 50])).split(area);

        let sparkline_top = Sparkline::default()
            .block(Block::new())
            .data(freqs.clone())
            .style(Style::default().fg(Color::LightRed).bg(Color::Black));

        sparkline_top.render(chunks[0], buf);

        let max_freq = freqs.iter().max().unwrap();
        let inverse_freqs: Vec<u64> = freqs.iter().copied().map(|f| max_freq - f).collect();

        let sparkline_bottom = Sparkline::default()
            .block(Block::new())
            .data(inverse_freqs)
            .style(Style::default().bg(Color::LightRed).fg(Color::Black));

        sparkline_bottom.render(chunks[1], buf);
    }
}

fn main() -> io::Result<()> {
    let host = cpal::default_host();
    let device = host.default_input_device().expect("No input device found");
    let config = device
        .default_input_config()
        .expect("could not get default input config")
        .config();

    let fft = Arc::new(Mutex::new(SimpleFft::new(config.sample_rate.0 as usize)));
    let fft_feeder = fft.clone();

    let stream = device
        .build_input_stream(
            &config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                if let Ok(mut fft) = fft_feeder.lock() {
                    fft.feed_samples(data);
                }
            },
            |err| eprintln!("Error: {}", err),
            None,
        )
        .expect("could not build input stream");

    stream.play().expect("could not play input stream");

    let mut terminal = ratatui::init();
    let app_result = App::new(fft).run(&mut terminal);
    ratatui::restore();

    app_result
}
