use caw::prelude::SignalPlayer;
use caw_core::ast::main::oscillator_expr;
use caw_core::ast::value::Value;

pub fn main() -> anyhow::Result<()> {
    let expr = oscillator_expr();
    let signal = expr.eval()?;

    println!("{}", signal.to_string());

    if let Value::Sf64(sf64_signal) = signal {
        let mut signal_player = SignalPlayer::new()?;
        signal_player.play_sample_forever(sf64_signal);
    } else {
        anyhow::bail!("Failed to evaluate cymbal expression as Sf64 signal.");
    }

    Ok(())
}
