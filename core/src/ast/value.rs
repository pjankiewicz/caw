use std::sync::Arc;

use caw_proc_macros::FromValue;
use crate::ast::expr::Terminal;
use crate::filters::{BandPassButterworth, BandPassChebyshev, Compress, Delay, DownSample, Echo, EnvelopeFollower, HighPassButterworth, HighPassChebyshev, LowPassButterworth, LowPassChebyshev, LowPassMoogLadder, NoiseGate, Quantize, QuantizeToScale, Reverb, SampleAndHold, Saturate};
use crate::loopers::ClockedMidiNoteMonophonicLooper;
use crate::oscillator::{Oscillator, Waveform};
use crate::prelude::*;

#[derive(FromValue)]
// represents evaluated expression
pub enum Value {
    Terminal(Terminal),
    Sfreq(Signal<Freq>),
    Sf64(Signal<f64>),
    Sf32(Signal<f32>),
    Su32(Signal<u32>),
    Sbool(Signal<bool>),
    Su8(Signal<u8>),
    SWaveform(Signal<Waveform>),
    Sample(Sample),
    LowPassButterworth(LowPassButterworth),
    HighPassButterworth(HighPassButterworth),
    LowPassMoogLadder(LowPassMoogLadder),
    Gate(Gate),
    Trigger(Trigger),
    Zip((Box<Value>, Box<Value>)),
    GateU8((Gate, Su8)),
    BandPassButterworth(BandPassButterworth),
    LowPassChebyshev(LowPassChebyshev),
    HighPassChebyshev(HighPassChebyshev),
    BandPassChebyshev(BandPassChebyshev),
    Saturate(Saturate),
    Compress(Compress),
    Delay(Delay),
    Echo(Echo),
    SampleAndHold(SampleAndHold),
    Quantize(Quantize),
    DownSample(DownSample),
    QuantizeToScale(QuantizeToScale),
    Reverb(Reverb),
    EnvelopeFollower(EnvelopeFollower),
    NoiseGate(NoiseGate),
    ClockedMidiNoteMonophonicLooper(ClockedMidiNoteMonophonicLooper),
    Oscillator(Oscillator),
}


impl Value {
    pub fn to_string(&self) -> String {
        match self {
            Value::Terminal(_) => "Terminal".to_string(),
            Value::Sfreq(_) => "Sfreq".to_string(),
            Value::Sf64(_) => "Sf64".to_string(),
            Value::Sf32(_) => "Sf32".to_string(),
            Value::Sbool(_) => "Sbool".to_string(),
            Value::SWaveform(_) => "Waveform".to_string(),
            Value::Gate(_) => "Gate".to_string(),
            Value::Trigger(_) => "Trigger".to_string(),
            Value::Su8(_) => "Su8".to_string(),
            Value::Zip(_) => "Zip".to_string(),
            Value::LowPassButterworth(_) => "LowPassButterworth".to_string(),
            Value::HighPassButterworth(_) => "HighPassButterworth".to_string(),
            Value::LowPassMoogLadder(_) => "LowPassMoogLadder".to_string(),
            Value::BandPassButterworth(_) => "BandPassButterworth".to_string(),
            Value::LowPassChebyshev(_) => "LowPassChebyshev".to_string(),
            Value::HighPassChebyshev(_) => "HighPassChebyshev".to_string(),
            Value::BandPassChebyshev(_) => "BandPassChebyshev".to_string(),
            Value::Saturate(_) => "Saturate".to_string(),
            Value::Compress(_) => "Compress".to_string(),
            Value::Delay(_) => "Delay".to_string(),
            Value::Echo(_) => "Echo".to_string(),
            Value::SampleAndHold(_) => "SampleAndHold".to_string(),
            Value::Quantize(_) => "Quantize".to_string(),
            Value::DownSample(_) => "DownSample".to_string(),
            Value::QuantizeToScale(_) => "QuantizeToScale".to_string(),
            Value::Reverb(_) => "Reverb".to_string(),
            Value::EnvelopeFollower(_) => "EnvelopeFollower".to_string(),
            Value::NoiseGate(_) => "NoiseGate".to_string(),
            Value::ClockedMidiNoteMonophonicLooper(_) => "ClockedMidiNoteMonophonicLooper".to_string(),
            Value::Oscillator(_) => "Oscillator".to_string(),
            Value::Sample(_) => "Sample".to_string(),
            Value::GateU8(_) => "GateU8".to_string(),
            Value::Su32(_) => "Su32".to_string()
        }
    }
}