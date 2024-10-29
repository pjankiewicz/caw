pub struct FMSynth {
    pub carrier_signal: Sf64,     // Carrier signal (in arbitrary units)
    pub modulator_signal: Sf64,   // Modulator signal (in arbitrary units)
    pub modulation_index: Sf64,   // Modulation index signal
}

impl FMSynth {
    pub fn signal(self) -> Sf64 {
        Signal::from_fn(move |ctx| {
            // Sample the carrier and modulator signals
            let carrier = self.carrier_signal.sample(ctx);
            let modulator = self.modulator_signal.sample(ctx);
            let modulation_index = self.modulation_index.sample(ctx);

            // Apply frequency modulation
            let modulated_carrier = carrier * (1.0 + modulation_index * modulator);

            modulated_carrier
        })
    }
}