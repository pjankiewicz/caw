use currawong::prelude::*;

fn run(signal: Sf64) -> anyhow::Result<()> {
    let mut signal_player = SignalPlayer::new()?;
    signal_player.play_sample_forever(signal);
}

fn main() -> anyhow::Result<()> {
    let gate = periodic_gate_s(
        oscillator_s(Waveform::Sine, 10.0)
            .reset_offset_01(0.5)
            .build()
            .signed_to_01()
            * 0.4
            + 0.02,
    )
    .duty_01(0.05)
    .build();
    let freq_hz = 100.0;
    let osc = oscillator_hz(Waveform::Pulse, freq_hz).build();
    let env = adsr_linear_01(&gate).release_s(0.1).build().exp_01(5.0);
    let volume_env = adsr_linear_01(gate).release_s(0.1).build();
    let signal = osc
        .filter(low_pass_moog_ladder(env * 6000.0).resonance(2.0).build())
        .mul_lazy(&volume_env);
    run(signal)
}
