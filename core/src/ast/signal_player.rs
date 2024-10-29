use std::time::{Duration, Instant};
use crate::{
    ast::sample_player::SamplePlayer,
    signal::{Signal, SignalCtx},
};

const SAFETY_VOLUME_THRESHOLD: f32 = 10.0;

pub struct SignalPlayer {
    sample_player: SamplePlayer,
    sample_index: u64,
}

pub trait ToF32 {
    fn to_f32(self) -> f32;
}

impl ToF32 for f32 {
    fn to_f32(self) -> f32 {
        self
    }
}

impl ToF32 for f64 {
    fn to_f32(self) -> f32 {
        self as f32
    }
}

impl SignalPlayer {
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self {
            sample_player: SamplePlayer::new()?,
            sample_index: 0,
        })
    }

    pub fn new_with_downsample(downsample: u32) -> anyhow::Result<Self> {
        Ok(Self {
            sample_player: SamplePlayer::new_with_downsample(downsample)?,
            sample_index: 0,
        })
    }

    pub fn send_signal_with_callback<
        T: Copy + Default + ToF32 + 'static,
        F: FnMut(f32),
    >(
        &mut self,
        signal: &mut Signal<T>,
        mut f: F,
    ) {
        let sample_rate_hz = self.sample_player.sample_rate_hz();
        self.sample_player.play_stream(|| {
            let ctx = SignalCtx {
                sample_index: self.sample_index,
                sample_rate_hz: sample_rate_hz as f64,
            };
            let sample = signal
                .sample(&ctx)
                .to_f32()
                .clamp(-SAFETY_VOLUME_THRESHOLD, SAFETY_VOLUME_THRESHOLD);
            f(sample);
            self.sample_index += 1;
            sample
        });
    }

    pub fn send_signal<T: Copy + Default + ToF32 + 'static>(
        &mut self,
        signal: &mut Signal<T>,
    ) {
        self.send_signal_with_callback(signal, |_| ());
    }

    #[cfg(not(feature = "web"))]
    pub fn play_sample_forever<T: Copy + Default + ToF32 + 'static>(
        &mut self,
        mut signal: Signal<T>,
    ) -> ! {
        use std::{thread, time::Duration};
        const PERIOD: Duration = Duration::from_millis(16);
        loop {
            self.send_signal(&mut signal);
            thread::sleep(PERIOD);
        }
    }

    pub fn set_volume(&self, volume: f32) {
        self.sample_player.set_volume(volume);
    }

    pub fn set_buffer_padding_sample_rate_ratio(
        &mut self,
        buffer_padding_sample_rate_ratio: f64,
    ) {
        self.sample_player.buffer_padding =
            (self.sample_player.sample_rate_hz() as f64
                * buffer_padding_sample_rate_ratio) as u64;
    }

    pub fn play_signal_for_duration<T: Copy + Default + ToF32 + 'static>(
        &mut self,
        mut signal: Signal<T>,
        duration_secs: f64,
    ) {
        let sample_rate_hz = self.sample_player.sample_rate_hz() as f64;
        let total_samples = (sample_rate_hz * duration_secs).round() as u64;

        let start_time = Instant::now();
        for _ in 0..total_samples {
            self.send_signal(&mut signal);
            // Optional sleep to reduce CPU usage, adjust as needed for smooth playback.
            std::thread::sleep(Duration::from_micros(10));
        }
        let elapsed = start_time.elapsed().as_secs_f64();
        println!("Played signal for {:.2} seconds (target was {:.2} seconds)", elapsed, duration_secs);
    }

    // Method specifically for playing for 1 second
    pub fn play_signal_for_one_second<T: Copy + Default + ToF32 + 'static>(
        &mut self,
        signal: Signal<T>,
    ) {
        self.play_signal_for_duration(signal, 1.0);
    }
}
