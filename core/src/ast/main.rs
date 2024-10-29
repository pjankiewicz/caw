use crate::ast::expr::{Expr, Terminal};
use crate::prelude::{Freq, Waveform};

fn cymbal_expr() -> Expr {
    let gate = Expr::PeriodicGate {
        freq: Box::new(Expr::Constant(Terminal::Freq(Freq::from_hz(0.25)))),
        duty_01: Some(Box::new(Expr::Constant(Terminal::F64(0.01)))),
        offset_01: Some(Box::new(Expr::Constant(Terminal::F64(0.0)))),
    };

    let osc = Expr::Noise;

    let env = Expr::ApplyFilter {
        signal: Box::new(Expr::AdsrLinear01 {
            key_down: Box::new(gate),
            key_press: None,
            attack_s: None,
            decay_s: None,
            sustain_01: None,
            release_s: Some(Box::new(Expr::Constant(Terminal::F64(0.1)))),
        }),
        filter: Box::new(Expr::LowPassButterworth {
            cutoff_hz: Box::new(Expr::Constant(Terminal::F64(100.0))),
            filter_order_half: None,
        }),
    };

    let filtered_osc = Expr::ApplyFilter {
        signal: Box::new(Expr::ApplyFilter {
            signal: Box::new(osc),
            filter: Box::new(Expr::LowPassMoogLadder {
                cutoff_hz: Box::new(Expr::Mul {
                    lhs: Box::new(Expr::Constant(Terminal::F64(10000.0))),
                    rhs: Box::new(env.clone())
                }),
                resonance: None,
            }),
        }),
        filter: Box::new(Expr::HighPassButterworth {
            cutoff_hz: Box::new(Expr::Constant(Terminal::F64(6000.0))),
            filter_order_half: None,
        }),
    };

    Expr::Mul{lhs: Box::new(filtered_osc), rhs: Box::new(env)}
}

pub fn oscillator_expr() -> Expr {
    Expr::Oscillator {
        waveform: Box::new(Expr::Constant(Terminal::Waveform(Waveform::Sine))),
        freq: Box::new(Expr::Constant(Terminal::Freq(Freq::from_hz(440.0)))),
        pulse_width_01: None,
        reset_trigger: None,
        reset_offset_01: None,
        hard_sync: None,
    }
}

// #[cfg(test)]
// mod tests {
//     use crate::ast::main::oscillator_expr;
//     use crate::ast::value::Value;
//
//     #[test]
//     fn test_sound() -> anyhow::Result<()> {
//         let expr = oscillator_expr();
//         let signal = expr.eval()?;
//
//         println!("{}", signal.to_string());
//
//         if let Value::Sf32(sf64_signal) = signal {
//             let mut signal_player = SignalPlayer::new()?;
//             signal_player.play_sample_forever(sf64_signal);
//         } else {
//             anyhow::bail!("Failed to evaluate cymbal expression as Sf64 signal.");
//         }
//
//         Ok(())
//     }
// }