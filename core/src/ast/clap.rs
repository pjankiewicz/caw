use crate::ast::expr::{Expr, Terminal};
use crate::prelude::*;

pub fn kick_expr() -> Expr {
    let gate = Expr::PeriodicGate {
        freq: Box::new(Expr::Constant(Terminal::Freq(Freq::from_hz(1.0)))),
        duty_01: Some(Box::new(Expr::Constant(Terminal::F64(0.01)))),
        offset_01: Some(Box::new(Expr::Constant(Terminal::F64(0.0)))),
    };

    let osc = Expr::Add {
        lhs: Box::new(Expr::Oscillator {
            waveform: Box::new(Expr::Constant(Terminal::Waveform(Waveform::Pulse))),
            freq: Box::new(Expr::Constant(Terminal::Freq(Freq::from_hz(80.0)))),
            pulse_width_01: None,
            reset_trigger: None,
            reset_offset_01: None,
            hard_sync: None,
        }),
        rhs: Box::new(Expr::Mul {
            lhs: Box::new(Expr::Noise),
            rhs: Box::new(Expr::Constant(Terminal::F64(1.5))),
        }),
    };

    let env = Expr::ApplyFilter {
        signal: Box::new(Expr::AdsrLinear01 {
            key_down: Box::new(gate.clone()),
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

    Expr::Mul{lhs: Box::new(mean_expr(osc, env.clone())), rhs: Box::new(env)}
}

pub fn snare_expr() -> Expr {
    let gate = Expr::PeriodicGate {
        freq: Box::new(Expr::Constant(Terminal::Freq(Freq::from_hz(2.0)))),
        duty_01: Some(Box::new(Expr::Constant(Terminal::F64(0.01)))),
        offset_01: Some(Box::new(Expr::Constant(Terminal::F64(0.0)))),
    };

    let osc = Expr::Noise;

    let env = Expr::ApplyFilter {
        signal: Box::new(Expr::AdsrLinear01 {
            key_down: Box::new(gate.clone()),
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

    let voice = Expr::ApplyFilter {
        signal: Box::new(osc),
        filter: Box::new(Expr::LowPassMoogLadder {
            cutoff_hz: Box::new(Expr::Mul {
                lhs: Box::new(Expr::Constant(Terminal::F64(5000.0))),
                rhs: Box::new(env.clone()),
            }),
            resonance: Some(Box::new(Expr::Constant(Terminal::F64(1.0)))),
        }),
    };

    Expr::Mul{lhs: Box::new(voice), rhs: Box::new(env)}
}

fn mean_expr(osc: Expr, env: Expr) -> Expr {
    Expr::Mean{exprs: vec![
        Box::new(Expr::Mul {
            lhs: Box::new(Expr::ApplyFilter {
                signal: Box::new(osc.clone()),
                filter: Box::new(Expr::LowPassMoogLadder {
                    cutoff_hz: Box::new(Expr::Mul {
                        lhs: Box::new(Expr::Constant(Terminal::F64(3000.0))),
                        rhs: Box::new(env.clone()),
                    }),
                    resonance: None,
                }),
            }),
            rhs: Box::new(Expr::Constant(Terminal::F64(0.5))),
        }),
        Box::new(Expr::ApplyFilter {
            signal: Box::new(osc.clone()),
            filter: Box::new(Expr::LowPassMoogLadder {
                cutoff_hz: Box::new(Expr::Mul {
                    lhs: Box::new(Expr::Constant(Terminal::F64(2000.0))),
                    rhs: Box::new(env.clone()),
                }),
                resonance: None,
            }),
        }),
        Box::new(Expr::ApplyFilter {
            signal: Box::new(osc.clone()),
            filter: Box::new(Expr::LowPassMoogLadder {
                cutoff_hz: Box::new(Expr::Mul {
                    lhs: Box::new(Expr::Constant(Terminal::F64(1000.0))),
                    rhs: Box::new(env.clone()),
                }),
                resonance: None,
            }),
        }),
        Box::new(Expr::Mul {
            lhs: Box::new(Expr::ApplyFilter {
                signal: Box::new(osc.clone()),
                filter: Box::new(Expr::LowPassMoogLadder {
                    cutoff_hz: Box::new(Expr::Mul {
                        lhs: Box::new(Expr::Constant(Terminal::F64(500.0))),
                        rhs: Box::new(env.clone()),
                    }),
                    resonance: None,
                }),
            }),
            rhs: Box::new(Expr::Constant(Terminal::F64(2.0))),
        }),
        Box::new(Expr::Mul{
            lhs: Box::new(Expr::ApplyFilter {
                signal: Box::new(osc),
                filter: Box::new(Expr::LowPassMoogLadder {
                    cutoff_hz: Box::new(Expr::Mul {
                        lhs: Box::new(Expr::Constant(Terminal::F64(250.0))),
                        rhs: Box::new(env.clone()),
                    }),
                    resonance: None,
                }),
            }),
            rhs: Box::new(Expr::Constant(Terminal::F64(2.0))),
        }),
    ]}
}