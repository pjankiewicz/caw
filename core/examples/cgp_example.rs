use std::collections::HashSet;
use rand::prelude::StdRng;
use rand::SeedableRng;
use caw_core::ast::cgp::CGPGenotype;
use caw_core::ast::drum_sounds::snare;
use caw_core::ast::expr::{Expr, Terminal};
use caw_core::ast::signal_player::SignalPlayer;
use caw_core::ast::value::Value;
use caw_core::oscillator::Waveform;
use caw_core::prelude::{adsr_linear_01, high_pass_butterworth, low_pass_butterworth, low_pass_moog_ladder, noise, periodic_gate_s, Sf64};
use caw_core::signal::{Freq, Sf32};

fn run(signal: Sf64) -> anyhow::Result<()> {
    let mut signal_player = SignalPlayer::new()?;
    signal_player.play_signal_for_one_second(signal);
    Ok(())
}

fn run32(signal: Sf32) -> anyhow::Result<()> {
    let mut signal_player = SignalPlayer::new()?;
    signal_player.play_signal_for_one_second(signal);
    Ok(())
}

fn cymbal() -> Sf64 {
    let gate = periodic_gate_s(0.25).duty_01(0.01).build();
    let osc = noise();
    let env = adsr_linear_01(gate)
        .release_s(0.1)
        .build()
        .filter(low_pass_butterworth(100.0).build());
    osc.filter(low_pass_moog_ladder(10000 * &env).build())
        .filter(high_pass_butterworth(6000.0).build())
        * env
}

fn main() {
    // Define inputs
    let inputs = vec![
        Expr::Constant(Terminal::Freq(Freq::from_hz(440.0))),
        Expr::Constant(Terminal::Freq(Freq::from_hz(880.0))),
        Expr::Constant(Terminal::Waveform(Waveform::Sine)),
    ];

    let function_set = Expr::function_set();

    println!("{:#?}", function_set);

    let max_arity = function_set.iter().map(|(_, _, arity, _)| arity.unwrap_or_default()).max().unwrap_or_default();
    let num_functions = function_set.len();

    let mut rng = StdRng::from_entropy();
    let mut genotype = CGPGenotype::new(
        &mut rng,
        3,
        1,
        max_arity,
        2,
        10,
        10,
        num_functions,
    );

    let mut last_expr = HashSet::new();

    for n in 0..1000000 {
        // println!("n - {}", n);
        genotype.mutate(&mut rng, 3);
        // println!("{:?}", genotype);

        // Build the expression from the genotype
        let expr = Expr::from_cgp_genotype(&genotype, inputs.clone());
        let signal = expr.eval();

        match signal {
            Ok(signal) => {
                // println!("{:?}", expr);
                if let Value::Sf64(signal) = &signal {
                    let expr_str = format!("{:#?}", expr);
                    if !last_expr.contains(&expr_str) {
                        println!("{}", expr_str);
                        let run_result = run(signal.clone());
                    }
                    last_expr.insert(expr_str);
                    // break;
                }
                if let Value::Sf32(signal) = &signal {
                    let expr_str = format!("{:#?}", expr);
                    if !last_expr.contains(&expr_str) {
                        println!("{}", expr_str);
                        let run_result = run32(signal.clone());
                    }
                    last_expr.insert(expr_str);
                    // break;
                }
            }
            Err(err) => {
                // println!("Expr {:?} Err {}", expr, err.to_string())
            }
        }
        // The expr should represent the Mean of certain expressions
    }

    println!("{:#?}", last_expr);
}