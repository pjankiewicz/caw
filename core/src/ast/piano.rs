use caw::prelude::*;
use crate::expr::{Expr, Terminal};

pub fn piano_expr() -> Expr {
    let gate = Expr::PeriodicGate {
        freq: Box::new(Expr::Constant(Terminal::Freq(Freq::from_hz(0.5)))),
        duty_01: Some(Box::new(Expr::Constant(Terminal::F64(0.1)))),
        offset_01: Some(Box::new(Expr::Constant(Terminal::F64(0.0)))),
    };

    let osc = Expr::Oscillator {
        waveform: Box::new(Expr::Constant(Terminal::Waveform(Waveform::Sine))),
        freq: Box::new(Expr::Constant(Terminal::Freq(Freq::from_hz(440.0)))),
        pulse_width_01: None,
        reset_trigger: None,
        reset_offset_01: None,
        hard_sync: None,
    };

    let env = Expr::AdsrLinear01 {
        key_down: Box::new(gate.clone()),
        key_press: None,
        attack_s: Some(Box::new(Expr::Constant(Terminal::F64(0.01)))),
        decay_s: Some(Box::new(Expr::Constant(Terminal::F64(0.2)))),
        sustain_01: Some(Box::new(Expr::Constant(Terminal::F64(0.8)))),
        release_s: Some(Box::new(Expr::Constant(Terminal::F64(0.3)))),
    };

    let filtered_osc = Expr::ApplyFilter {
        signal: Box::new(osc),
        filter: Box::new(Expr::LowPassButterworth {
            cutoff_hz: Box::new(Expr::Add(
                Box::new(Expr::Constant(Terminal::F64(1000.0))),
                Box::new(Expr::Mul(
                    Box::new(Expr::Constant(Terminal::F64(800.0))),
                    Box::new(env.clone()),
                )),
            )),
            filter_order_half: None,
        }),
    };

    Expr::Mul(Box::new(filtered_osc), Box::new(env))
}
