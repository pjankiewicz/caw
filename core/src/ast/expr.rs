use std::collections::HashMap;
use anyhow::anyhow;
use caw_proc_macros::FunctionSet;
use crate::ast::cgp::CGPGenotype;
use crate::ast::expr::Expr::{Saturate};
use crate::ast::value::Value;
use crate::builder::env::AdsrLinear01Builder;
use crate::builder::filter::{BandPassButterworthBuilder, BandPassChebyshevBuilder, CompressBuilder, DelayBuilder, EchoBuilder, EnvelopeFollowerBuilder, HighPassButterworthBuilder, HighPassChebyshevBuilder, LowPassButterworthBuilder, LowPassChebyshevBuilder, LowPassMoogLadderBuilder, NoiseGateBuilder, ReverbBuilder, SaturateBuilder};
use crate::builder::gate::PeriodicGateBuilder;
use crate::builder::loopers::{ClockedMidiNoteMonophonicLooperBuilder, ClockedTriggerLooperBuilder};
use crate::builder::oscillator::{OscillatorBuilder, OscillatorSimpleBuilder};
use crate::builder::patches::PulsePwmBuilder;
use crate::builder::sampler::SamplerBuilder;
use crate::filters::{DownSample, Quantize, QuantizeToScale, SampleAndHold};
use crate::prelude::*;
use crate::signal::{Sf32, Su32};

#[derive(Clone, Debug)]
pub enum Terminal {
    F64(f64),
    F32(f32),
    U32(u32),
    U8(u8),
    Bool(bool),
    Freq(Freq),
    Waveform(Waveform)
}

#[derive(Clone, Debug, FunctionSet)]
pub enum Expr {
    #[exclude_from_cgp]
    Constant(Terminal),
    Noise,
    #[exclude_from_cgp]
    Noise01,
    #[exclude_from_cgp]
    Zip{lhs: Box<Expr>, rhs: Box<Expr>},
    #[exclude_from_cgp]
    Mean{exprs: Vec<Box<Expr>>},
    Add{lhs: Box<Expr>, rhs: Box<Expr>},
    #[exclude_from_cgp]
    Sub{lhs: Box<Expr>, rhs: Box<Expr>},
    #[exclude_from_cgp]
    Mul{lhs: Box<Expr>, rhs: Box<Expr>},
    #[exclude_from_cgp]
    Div{lhs: Box<Expr>, rhs: Box<Expr>},
    #[exclude_from_cgp]
    ApplyFilter{signal: Box<Expr>, filter: Box<Expr>},
    #[exclude_from_cgp]
    PeriodicGate{freq: Box<Expr>, duty_01: Option<Box<Expr>>, offset_01: Option<Box<Expr>>},
    #[exclude_from_cgp]
    AdsrLinear01{key_down: Box<Expr>, key_press: Option<Box<Expr>>, attack_s: Option<Box<Expr>>, decay_s: Option<Box<Expr>>, sustain_01: Option<Box<Expr>>, release_s: Option<Box<Expr>>},
    #[exclude_from_cgp]
    LowPassButterworth{cutoff_hz: Box<Expr>, filter_order_half: Option<usize>},
    #[exclude_from_cgp]
    HighPassButterworth{cutoff_hz: Box<Expr>, filter_order_half: Option<usize>},
    #[exclude_from_cgp]
    BandPassButterworth{cutoff_hz_lower: Box<Expr>, cutoff_hz_upper: Box<Expr>, filter_order_quarter: Option<usize>},
    #[exclude_from_cgp]
    LowPassChebyshev{cutoff_hz: Box<Expr>, resonance: Option<Box<Expr>>, filter_order_half: Option<usize>},
    #[exclude_from_cgp]
    HighPassChebyshev{cutoff_hz: Box<Expr>, resonance: Option<Box<Expr>>, filter_order_half: Option<usize>},
    #[exclude_from_cgp]
    BandPassChebyshev{cutoff_hz_lower: Box<Expr>, cutoff_hz_upper: Box<Expr>, resonance: Option<Box<Expr>>, filter_order_quarter: Option<usize>},
    #[exclude_from_cgp]
    LowPassMoogLadder{cutoff_hz: Box<Expr>, resonance: Option<Box<Expr>>},
    #[exclude_from_cgp]
    Saturate{scale: Option<Box<Expr>>, min: Option<Box<Expr>>, max: Option<Box<Expr>>},
    #[exclude_from_cgp]
    Compress{threshold: Option<Box<Expr>>, ratio: Option<Box<Expr>>, scale: Option<Box<Expr>>},
    #[exclude_from_cgp]
    Delay{time_s: Option<Box<Expr>>},
    #[exclude_from_cgp]
    Echo{time_s: Option<Box<Expr>>, scale: Option<Box<Expr>>},
    #[exclude_from_cgp]
    SampleAndHold{trigger: Box<Expr>},
    #[exclude_from_cgp]
    Quantize{resolution: Box<Expr>},
    #[exclude_from_cgp]
    DownSample{scale: Box<Expr>},
    #[exclude_from_cgp]
    QuantizeToScale{notes: Vec<Box<Expr>>},
    #[exclude_from_cgp]
    Reverb{damping: Option<Box<Expr>>, room_size: Option<Box<Expr>>},
    #[exclude_from_cgp]
    EnvelopeFollower{sensitivity_hz: Option<Box<Expr>>},
    #[exclude_from_cgp]
    NoiseGate{control: Box<Expr>, threshold: Option<Box<Expr>>, ratio: Option<Box<Expr>>},
    #[exclude_from_cgp]
    ClockedTriggerLooper{clock: Option<Box<Expr>>, add: Option<Box<Expr>>, remove: Option<Box<Expr>>, length: Option<usize>},
    #[exclude_from_cgp]
    ClockedMidiNoteMonophonicLooper{clock: Option<Box<Expr>>, input_gate: Option<Box<Expr>>, input_midi_index: Option<Box<Expr>>, clear: Option<Box<Expr>>, length: Option<usize>},
    #[exclude_from_cgp]
    Oscillator{waveform: Box<Expr>, freq: Box<Expr>, pulse_width_01: Option<Box<Expr>>, reset_trigger: Option<Box<Expr>>, reset_offset_01: Option<Box<Expr>>, hard_sync: Option<Box<Expr>>},
    OscillatorSimple{waveform: Box<Expr>, freq: Box<Expr>},
    // Sampler{sample: Box<Expr>, trigger: Option<Box<Expr>>},

    // Gate conversions
    /// Source: Gate::to_signal
    #[exclude_from_cgp]
    GateToSignal { gate: Box<Expr> },

    /// Source: Gate::to_01
    #[exclude_from_cgp]
    GateToSignal01 { gate: Box<Expr> },

    /// Source: Gate::to_trigger_rising_edge
    #[exclude_from_cgp]
    GateToTriggerRisingEdge { gate: Box<Expr> },

    // Trigger conversions
    /// Source: Trigger::to_signal
    #[exclude_from_cgp]
    TriggerToSignal { trigger: Box<Expr> },

    /// Source: Trigger::to_gate
    #[exclude_from_cgp]
    TriggerToGate { trigger: Box<Expr> },

    /// Source: Trigger::to_gate_with_duration_s
    #[exclude_from_cgp]
    TriggerToGateWithDurationS { trigger: Box<Expr>, duration_s: Box<Expr> },

    /// Source: Trigger::any
    #[exclude_from_cgp]
    TriggerAny { triggers: Vec<Box<Expr>> },

    /// Source: Trigger::divide
    #[exclude_from_cgp]
    TriggerDivide { trigger: Box<Expr>, divisor: Box<Expr> },

    /// Source: Trigger::random_skip
    #[exclude_from_cgp]
    TriggerRandomSkip { trigger: Box<Expr>, probability_01: Box<Expr> },

    // Signal<bool> conversions
    /// Source: Signal<bool>::to_trigger_raw
    #[exclude_from_cgp]
    SignalBoolToTriggerRaw { signal: Box<Expr> },

    /// Source: Signal<bool>::to_gate
    #[exclude_from_cgp]
    SignalBoolToGate { signal: Box<Expr> },

    // Signal<f64> conversions and transformations
    /// Source: Signal<f64>::lazy_zero
    #[exclude_from_cgp]
    SignalF64LazyZero { signal: Box<Expr>, control: Box<Expr> },

    /// Source: Signal<f64>::mul_lazy
    #[exclude_from_cgp]
    SignalF64MulLazy { signal: Box<Expr>, multiplier: Box<Expr> },

    /// Source: Signal<f64>::force_lazy
    #[exclude_from_cgp]
    SignalF64ForceLazy { signal: Box<Expr>, other: Box<Expr> },

    /// Source: Signal<f64>::exp_01
    #[exclude_from_cgp]
    SignalF64Exp01 { signal: Box<Expr>, k: Box<Expr> },

    /// Source: Signal<f64>::inv_01
    #[exclude_from_cgp]
    SignalF64Inv01 { signal: Box<Expr> },

    /// Source: Signal<f64>::signed_to_01
    #[exclude_from_cgp]
    SignalF64SignedTo01 { signal: Box<Expr> },

    /// Source: Signal<f64>::clamp_non_negative
    #[exclude_from_cgp]
    SignalF64ClampNonNegative { signal: Box<Expr> },

    /// Source: Signal<f64>::min
    #[exclude_from_cgp]
    SignalF64Min { lhs: Box<Expr>, rhs: Box<Expr> },

    /// Source: Signal<f64>::max
    #[exclude_from_cgp]
    SignalF64Max { lhs: Box<Expr>, rhs: Box<Expr> },

    // Additional conversions on Signal<bool>, Gate, and Trigger to transform or combine
    /// Source: Signal<Option<T>>::or
    // SignalOptionOr { lhs: Box<Expr>, rhs: Box<Expr> },

    /// Source: Signal::then
    // SignalThen { signal: Box<Expr>, function: Box<Expr> },

    /// Source: Signal::unzip (tuple of 2)
    // SignalUnzip2 { signal: Box<Expr> },

    /// Source: Signal::unzip (tuple of 3)
    // SignalUnzip3 { signal: Box<Expr> },

    // Signal<Freq> specific conversions
    /// Source: Signal<Freq>::hz
    #[exclude_from_cgp]
    SignalFreqToHz { signal: Box<Expr> },

    /// Source: Signal<Freq>::s
    #[exclude_from_cgp]
    SignalFreqToS { signal: Box<Expr> },

    // Signal<u8> conversions
    /// Source: Signal<u8>::midi_index_to_freq_hz_a440
    #[exclude_from_cgp]
    SignalU8ToFreqHz { midi_index: Box<Expr> },

    /// Source: Signal<u8>::midi_index_to_freq_a440
    #[exclude_from_cgp]
    SignalU8ToFreq { midi_index: Box<Expr> },
}

impl Expr {
    pub fn from_cgp_genotype(genotype: &CGPGenotype, inputs: Vec<Expr>) -> Expr {
        let mut node_outputs = inputs.clone();

        // Get the function set mapping
        let function_set = Self::function_set();

        // Create a map from function indices to their data
        let function_map: HashMap<usize, (String, Option<usize>, fn(&[Expr], &mut usize) -> Expr)> =
            function_set
                .into_iter()
                .map(|(name, idx, arity, constructor)| (idx, (name, arity, constructor)))
                .collect();

        // Process nodes
        for node_idx in 0..genotype.num_nodes {
            let f_idx = genotype.function_genes[node_idx];
            let (name, arity_opt, constructor) = match function_map.get(&f_idx) {
                Some(&(ref name, arity, constructor)) => (name, arity, constructor),
                None => panic!("Invalid function index {}", f_idx),
            };

            let input_indices = &genotype.connection_genes[node_idx];

            // println!("{} {:?} {:?}", name, arity_opt, genotype.connection_genes);

            // Collect input expressions based on arity
            let input_exprs = if let Some(arity) = arity_opt {
                // Ensure we have enough inputs
                if input_indices.len() < arity {
                    panic!(
                        "Node {} ({}) requires {} inputs, but only {} provided.",
                        node_idx,
                        name,
                        arity,
                        input_indices.len()
                    );
                }
                // Take only the number of inputs required by the function's arity
                input_indices
                    .iter()
                    .take(arity)
                    .map(|&idx| {
                        if idx >= node_outputs.len() {
                            panic!("Invalid connection index {} at node {}", idx, node_idx);
                        }
                        node_outputs[idx].clone()
                    })
                    .collect::<Vec<Expr>>()
            } else {
                // Variable arity function: use all input indices
                input_indices
                    .iter()
                    .map(|&idx| {
                        if idx >= node_outputs.len() {
                            panic!("Invalid connection index {} at node {}", idx, node_idx);
                        }
                        node_outputs[idx].clone()
                    })
                    .collect::<Vec<Expr>>()
            };

            let mut input_idx = 0;
            let expr = constructor(&input_exprs, &mut input_idx);

            node_outputs.push(expr);
        }

        // Collect output expressions
        let output_exprs: Vec<Expr> = genotype
            .output_genes
            .iter()
            .map(|&idx| {
                if idx >= node_outputs.len() {
                    panic!("Invalid output index {}", idx);
                }
                node_outputs[idx].clone()
            })
            .collect();

        // Return the final expression
        if output_exprs.len() == 1 {
            output_exprs.into_iter().next().unwrap()
        } else {
            Expr::Mean {
                exprs: output_exprs.into_iter().map(Box::new).collect(),
            }
        }
    }
}


pub trait FromExpr {
    fn from_expr(expr: &Expr) -> Result<Value, anyhow::Error>
        where
            Self: Sized;
}

impl Expr {
    pub fn eval(&self) -> anyhow::Result<Value> {
        match self {
            Expr::Constant(value) => {
                match value {
                    Terminal::F64(v) => Ok(const_::<f64>(v.clone()).into()),
                    Terminal::F32(v) => Ok(const_::<f32>(v.clone()).into()),
                    Terminal::U8(v) => Ok(const_::<u8>(v.clone()).into()),
                    Terminal::Bool(v) => Ok(const_::<bool>(v.clone()).into()),
                    Terminal::Freq(v) => Ok(const_::<Freq>(v.clone()).into()),
                    Terminal::Waveform(v) => Ok(const_::<Waveform>(v.clone()).into()),
                    Terminal::U32(v) => Ok(const_::<u32>(v.clone()).into())
                }
            }
            Expr::Noise => Ok(noise().into()),
            Expr::Noise01 => Ok(noise_01().into()),
            Expr::Zip{lhs, rhs} => {
                let lhs = Box::new(lhs.eval()?);
                let rhs = Box::new(rhs.eval()?);
                Ok(Value::Zip((lhs, rhs)))
            }
            Expr::Mean{exprs: signals} => {
                let mut signals_eval = vec![];
                for signal in signals {
                    let signal_eval = signal.eval()?;
                    signals_eval.push(signal_eval);
                }
                let signals_f64: Vec<_> = signals_eval.iter().filter_map(|s| {
                    match s {
                        Value::Sf64(s) => Some(s.clone()),
                        _ => None
                    }
                }).collect();
                Ok(mean(signals_f64).into())
            }
            Expr::Add{lhs, rhs} => {
                let lhs_value = lhs.eval()?;
                let lhs_value_str = lhs_value.to_string();
                let rhs_value = rhs.eval()?;
                let rhs_value_str = rhs_value.to_string();
                match (lhs_value, rhs_value) {
                    (Value::Sf32(lhs), Value::Sf32(rhs)) => Ok(Value::Sf32((lhs + rhs).into())),
                    (Value::Sf64(lhs), Value::Sf64(rhs)) => Ok(Value::Sf32((lhs+ rhs).into())),
                    (Value::Sf32(lhs), Value::Sf64(rhs)) => Ok(Value::Sf32((lhs + Sf32::from(rhs)).into())),
                    (Value::Sf64(lhs), Value::Sf32(rhs)) => Ok(Value::Sf32((Sf32::from(lhs) + rhs).into())),
                    (_, _) => Err(anyhow!("Error evaluating {:?} + {:?}", lhs_value_str, rhs_value_str))
                }
            }
            Expr::Sub{lhs, rhs} => {
                let lhs_value = lhs.eval()?;
                let rhs_value = rhs.eval()?;
                match (lhs_value, rhs_value) {
                    (Value::Sf32(lhs), Value::Sf32(rhs)) => Ok(Value::Sf32((lhs - rhs).into())),
                    (Value::Sf64(lhs), Value::Sf64(rhs)) => Ok(Value::Sf32((lhs - rhs).into())),
                    (Value::Sf32(lhs), Value::Sf64(rhs)) => Ok(Value::Sf32((lhs - Sf32::from(rhs)).into())),
                    (Value::Sf64(lhs), Value::Sf32(rhs)) => Ok(Value::Sf32((Sf32::from(lhs) - rhs).into())),
                    (_, _) => Err(anyhow!("Error evaluating {:?} - {:?}", lhs, rhs))
                }
            }
            Expr::Mul{lhs, rhs} => {
                let lhs_value = lhs.eval()?;
                let rhs_value = rhs.eval()?;
                match (lhs_value, rhs_value) {
                    (Value::Sf32(lhs), Value::Sf32(rhs)) => Ok(Value::Sf32((lhs * rhs).into())),
                    (Value::Sf64(lhs), Value::Sf64(rhs)) => Ok(Value::Sf32((lhs * rhs).into())),
                    (Value::Sf32(lhs), Value::Sf64(rhs)) => Ok(Value::Sf32((lhs * Sf32::from(rhs)).into())),
                    (Value::Sf64(lhs), Value::Sf32(rhs)) => Ok(Value::Sf32((Sf32::from(lhs) * rhs).into())),
                    (_, _) => Err(anyhow!("Error evaluating {:?} * {:?}", lhs, rhs))
                }
            },
            Expr::Div{lhs, rhs} => {
                let lhs_value = lhs.eval()?;
                let rhs_value = rhs.eval()?;
                match (lhs_value, rhs_value) {
                    (Value::Sf32(lhs), Value::Sf32(rhs)) => Ok(Value::Sf32((lhs / rhs).into())),
                    (Value::Sf64(lhs), Value::Sf64(rhs)) => Ok(Value::Sf32((lhs / rhs).into())),
                    (Value::Sf32(lhs), Value::Sf64(rhs)) => Ok(Value::Sf32((lhs / Sf32::from(rhs)).into())),
                    (Value::Sf64(lhs), Value::Sf32(rhs)) => Ok(Value::Sf32((Sf32::from(lhs) / rhs).into())),
                    (_, _) => Err(anyhow!("Error evaluating {:?} / {:?}", lhs, rhs))
                }
            }
            Expr::ApplyFilter { filter, signal } => {
                let filter_value = filter.eval()?;
                let filter_value_str = filter_value.to_string();
                let signal_value = signal.eval()?;
                let signal_value_str = signal_value.to_string();
                match (signal_value, filter_value) {
                    (Value::Sf64(signal), Value::HighPassButterworth(filter)) => {
                        Ok(Value::Sf64(signal.filter(filter)))
                    },
                    (Value::Sf64(signal), Value::LowPassButterworth(filter)) => {
                        Ok(Value::Sf64(signal.filter(filter)))
                    },
                    (Value::Sf64(signal), Value::LowPassMoogLadder(filter)) => {
                        Ok(Value::Sf64(signal.filter(filter)))
                    },
                    _ => Err(anyhow!("Could not apply filter {:?} to {:?}", filter_value_str, signal_value_str))
                }
            }

            // builders
            Expr::PeriodicGate { .. } => PeriodicGateBuilder::from_expr(&self.clone()),
            Expr::AdsrLinear01 { .. } => AdsrLinear01Builder::from_expr(&self.clone()),
            Expr::LowPassButterworth { .. } => LowPassButterworthBuilder::from_expr(&self.clone()),
            Expr::HighPassButterworth { .. } => HighPassButterworthBuilder::from_expr(&self.clone()),
            Expr::LowPassMoogLadder { .. } => LowPassMoogLadderBuilder::from_expr(&self.clone()),
            Expr::Oscillator { .. } => OscillatorBuilder::from_expr(&self.clone()),
            Expr::OscillatorSimple {.. } => OscillatorSimpleBuilder::from_expr(&self.clone()),
            Expr::BandPassButterworth { .. } => BandPassButterworthBuilder::from_expr(&self.clone()),
            Expr::LowPassChebyshev { .. } => LowPassChebyshevBuilder::from_expr(&self.clone()),
            Expr::HighPassChebyshev { .. } => HighPassChebyshevBuilder::from_expr(&self.clone()),
            Expr::BandPassChebyshev { .. } => BandPassChebyshevBuilder::from_expr(&self.clone()),
            Expr::Saturate { .. } => SaturateBuilder::from_expr(&self.clone()),
            Expr::Compress { .. } => CompressBuilder::from_expr(&self.clone()),
            Expr::Delay { .. } => DelayBuilder::from_expr(&self.clone()),
            Expr::Echo { .. } => EchoBuilder::from_expr(&self.clone()),
            Expr::Reverb { .. } => ReverbBuilder::from_expr(&self.clone()),
            Expr::EnvelopeFollower { sensitivity_hz } => EnvelopeFollowerBuilder::from_expr(&self.clone()),
            Expr::NoiseGate { .. } => NoiseGateBuilder::from_expr(&self.clone()),
            Expr::ClockedTriggerLooper { .. } => ClockedTriggerLooperBuilder::from_expr(&self.clone()),
            Expr::ClockedMidiNoteMonophonicLooper { .. } => ClockedMidiNoteMonophonicLooperBuilder::from_expr(&self.clone()),
            // Expr::Sampler { .. } => {todo!()},

            // filters
            Expr::SampleAndHold { trigger } => {
                let trigger: Trigger = trigger.eval()?.try_into().map_err(|_| anyhow!("Could not convert to Trigger"))?;
                Ok(SampleAndHold::new(trigger).into())
            }
            Expr::Quantize { resolution } => {
                let resolution: Sf64 = resolution.eval()?.try_into().map_err(|_| anyhow!("Could not convert to Sf64"))?;
                Ok(Quantize::new(resolution).into())
            }
            Expr::DownSample { scale } => {
                let scale: Sf64 = scale.eval()?.try_into().map_err(|_| anyhow!("Could not convert to Sf64"))?;
                Ok(DownSample::new(scale).into())
            }
            Expr::QuantizeToScale { notes } => {
                let notes: Vec<Sfreq> = notes.iter().map(|n| n.eval()?.try_into().map_err(|_| anyhow!("Could not convert to Sfreq"))).collect::<Result<_, _>>()?;
                Ok(QuantizeToScale::new(notes).into())
            }

            // GPT: implement all of those
            // Expr::GateToSignal { .. } => {todo!()}
            Expr::GateToSignal { gate } => {
                let gate: Gate = gate.eval()?.try_into().map_err(|_| anyhow!("Could not convert to Gate"))?;
                Ok(gate.to_signal().into())
            }
            Expr::GateToSignal01 { gate } => { let gate: Gate = gate.eval()?.try_into().map_err(|_| anyhow!("Could not convert to Gate"))?; Ok(gate.to_01().into()) }
            Expr::GateToTriggerRisingEdge { gate } => { let gate: Gate = gate.eval()?.try_into().map_err(|_| anyhow!("Could not convert to Gate"))?; Ok(gate.to_trigger_rising_edge().into()) }
            Expr::TriggerToSignal { trigger } => { let trigger: Trigger = trigger.eval()?.try_into().map_err(|_| anyhow!("Could not convert to Trigger"))?; Ok(trigger.to_signal().into()) }
            Expr::TriggerToGate { trigger } => { let trigger: Trigger = trigger.eval()?.try_into().map_err(|_| anyhow!("Could not convert to Trigger"))?; Ok(trigger.to_gate().into()) }
            Expr::TriggerToGateWithDurationS { trigger, duration_s } => { let trigger: Trigger = trigger.eval()?.try_into().map_err(|_| anyhow!("Could not convert to Trigger"))?; let duration_s: Sf64 = duration_s.eval()?.try_into().map_err(|_| anyhow!("Could not convert to Sf64"))?; Ok(trigger.to_gate_with_duration_s(duration_s).into()) }
            Expr::TriggerAny { triggers } => { let triggers: Vec<Trigger> = triggers.into_iter().map(|t| t.eval()?.try_into().map_err(|_| anyhow!("Could not convert to Trigger"))).collect::<Result<_, _>>()?; Ok(Trigger::any(triggers).into()) }
            Expr::TriggerDivide { trigger, divisor } => { let trigger: Trigger = trigger.eval()?.try_into().map_err(|_| anyhow!("Could not convert to Trigger"))?; let divisor: Su32 = divisor.eval()?.try_into().map_err(|_| anyhow!("Could not convert to Sf64"))?; Ok(trigger.divide(divisor).into()) }
            Expr::TriggerRandomSkip { trigger, probability_01 } => { let trigger: Trigger = trigger.eval()?.try_into().map_err(|_| anyhow!("Could not convert to Trigger"))?; let probability_01: Sf64 = probability_01.eval()?.try_into().map_err(|_| anyhow!("Could not convert to Sf64"))?; Ok(trigger.random_skip(probability_01).into()) }
            Expr::SignalBoolToTriggerRaw { signal } => { let signal_bool: Signal<bool> = signal.eval()?.try_into().map_err(|_| anyhow!("Could not convert to Signal<bool>"))?; Ok(signal_bool.to_trigger_raw().into()) }
            Expr::SignalBoolToGate { signal } => { let signal_bool: Signal<bool> = signal.eval()?.try_into().map_err(|_| anyhow!("Could not convert to Signal<bool>"))?; Ok(signal_bool.to_gate().into()) }
            Expr::SignalF64LazyZero { signal, control } => { let signal_f64: Signal<f64> = signal.eval()?.try_into().map_err(|_| anyhow!("Could not convert to Signal<f64>"))?; let control_f64: Sf64 = control.eval()?.try_into().map_err(|_| anyhow!("Could not convert to Sf64"))?; Ok(signal_f64.lazy_zero(&control_f64).into()) }
            Expr::SignalF64MulLazy { signal, multiplier } => { let signal_f64: Signal<f64> = signal.eval()?.try_into().map_err(|_| anyhow!("Could not convert to Signal<f64>"))?; let multiplier: Sf64 = multiplier.eval()?.try_into().map_err(|_| anyhow!("Could not convert to Sf64"))?; Ok(signal_f64.mul_lazy(&multiplier).into()) }
            Expr::SignalF64ForceLazy { signal, other } => { let signal_f64: Signal<f64> = signal.eval()?.try_into().map_err(|_| anyhow!("Could not convert to Signal<f64>"))?; let other_signal: Sf64 = other.eval()?.try_into().map_err(|_| anyhow!("Could not convert to Sf64"))?; Ok(signal_f64.force_lazy(&other_signal).into()) }
            Expr::SignalF64Exp01 { signal, k } => { let signal_f64: Signal<f64> = signal.eval()?.try_into().map_err(|_| anyhow!("Could not convert to Signal<f64>"))?; let k_value: Sf64 = k.eval()?.try_into().map_err(|_| anyhow!("Could not convert to Sf64"))?; Ok(signal_f64.exp_01(k_value).into()) }
            Expr::SignalF64Inv01 { signal } => { let signal_f64: Signal<f64> = signal.eval()?.try_into().map_err(|_| anyhow!("Could not convert to Signal<f64>"))?; Ok(signal_f64.inv_01().into()) }
            Expr::SignalF64SignedTo01 { signal } => { let signal_f64: Signal<f64> = signal.eval()?.try_into().map_err(|_| anyhow!("Could not convert to Signal<f64>"))?; Ok(signal_f64.signed_to_01().into()) }
            Expr::SignalF64ClampNonNegative { signal } => { let signal_f64: Signal<f64> = signal.eval()?.try_into().map_err(|_| anyhow!("Could not convert to Signal<f64>"))?; Ok(signal_f64.clamp_non_negative().into()) }
            Expr::SignalF64Min { lhs, rhs } => { let lhs_signal: Signal<f64> = lhs.eval()?.try_into().map_err(|_| anyhow!("Could not convert to Signal<f64>"))?; let rhs_signal: Signal<f64> = rhs.eval()?.try_into().map_err(|_| anyhow!("Could not convert to Signal<f64>"))?; Ok(lhs_signal.min(rhs_signal).into()) }
            Expr::SignalF64Max { lhs, rhs } => { let lhs_signal: Signal<f64> = lhs.eval()?.try_into().map_err(|_| anyhow!("Could not convert to Signal<f64>"))?; let rhs_signal: Signal<f64> = rhs.eval()?.try_into().map_err(|_| anyhow!("Could not convert to Signal<f64>"))?; Ok(lhs_signal.max(rhs_signal).into()) }
            Expr::SignalFreqToHz { signal } => { let freq_signal: Signal<Freq> = signal.eval()?.try_into().map_err(|_| anyhow!("Could not convert to Signal<Freq>"))?; Ok(freq_signal.hz().into()) }
            Expr::SignalFreqToS { signal } => { let freq_signal: Signal<Freq> = signal.eval()?.try_into().map_err(|_| anyhow!("Could not convert to Signal<Freq>"))?; Ok(freq_signal.s().into()) }
            Expr::SignalU8ToFreqHz { midi_index } => { let midi_index_signal: Signal<u8> = midi_index.eval()?.try_into().map_err(|_| anyhow!("Could not convert to Signal<u8>"))?; Ok(midi_index_signal.midi_index_to_freq_hz_a440().into()) }
            Expr::SignalU8ToFreq { midi_index } => { let midi_index_signal: Signal<u8> = midi_index.eval()?.try_into().map_err(|_| anyhow!("Could not convert to Signal<u8>"))?; Ok(midi_index_signal.midi_index_to_freq_a440().into()) }
            // those would demand implementation for all types?
            // Expr::SignalOptionOr { lhs, rhs } => { todo!() }
            // Expr::SignalThen { signal, function } => { todo!() }
            // Expr::SignalUnzip2 { signal } => { todo!() }
            // Expr::SignalUnzip3 { signal } => { todo!() }
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use super::*;

    #[test]
    fn test_functionset() {
        let function_set = Expr::function_set();
        println!("{:#?}", function_set);
    }

    #[test]
    fn test_genotype_to_expr() {
        // Define inputs
        let inputs = vec![
            Expr::Constant(Terminal::F64(1.0)),
            Expr::Constant(Terminal::F64(2.0)),
            Expr::Constant(Terminal::F64(3.0)),
        ];

        let function_set = Expr::function_set();

        println!("{:#?}", function_set);

        let max_arity = function_set.iter().map(|(_, _, arity, _)| arity.unwrap_or_default()).max().unwrap_or_default();
        let num_functions = function_set.len();

        let mut rng = StdRng::from_entropy();

        let mut genotype = CGPGenotype::new(
            &mut rng,
            3,
            10,
            max_arity,
            5,
            5,
            10,
            num_functions,
        );

        // genotype.mutate(&mut rng, 5);

        // Build the expression from the genotype
        let expr = Expr::from_cgp_genotype(&genotype, inputs);

        let signal = expr.eval().unwrap();

        // The expr should represent the Mean of certain expressions
        println!("{:#?}", expr);
    }
}