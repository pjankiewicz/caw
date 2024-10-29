use crate::ast::expr::{Expr, Terminal};
use crate::ast::value::Value;
use crate::oscillator::Waveform;
use crate::prelude::{Gate, Sample, Sf64, Sfreq, Signal, Su8, Trigger};
use crate::signal::{const_, Freq, Sbool, Su32};


impl TryFrom<Value> for Sf64 {
    type Error = ();

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::Terminal(v) => v.try_into(),
            Value::Sfreq(_) | Value::Sbool(_) | Value::Su8(_) | Value::SWaveform(_) | Value::Gate(_) 
            | Value::Trigger(_) | Value::Zip(_) | Value::BandPassButterworth(_) | Value::LowPassChebyshev(_)
            | Value::HighPassChebyshev(_) | Value::BandPassChebyshev(_) | Value::Saturate(_) 
            | Value::Compress(_) | Value::Delay(_) | Value::Echo(_) | Value::SampleAndHold(_)
            | Value::Quantize(_) | Value::DownSample(_) | Value::QuantizeToScale(_) | Value::Reverb(_)
            | Value::EnvelopeFollower(_) | Value::NoiseGate(_)
            | Value::ClockedMidiNoteMonophonicLooper(_) | Value::Oscillator(_) => Err(()),
            Value::Sf64(v) => Ok(v.into()),
            Value::Sf32(v) => Ok(v.into()),
            Value::LowPassButterworth(v) => Err(()),
            Value::HighPassButterworth(v) => Err(()),
            Value::LowPassMoogLadder(v) => Err(()),
            Value::Sample(_) => Err(()),
            Value::GateU8(_) => Err(()),
            Value::Su32(_) => Err(())
        }
    }
}

impl TryFrom<Value> for Gate {
    type Error = ();

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::Terminal(_) | Value::Sfreq(_) | Value::Sf64(_) | Value::Sf32(_) 
            | Value::Sbool(_) | Value::Su8(_) | Value::Trigger(_) | Value::Zip(_)
            | Value::LowPassButterworth(_) | Value::HighPassButterworth(_)
            | Value::LowPassMoogLadder(_) | Value::SWaveform(_)
            | Value::BandPassButterworth(_) | Value::LowPassChebyshev(_)
            | Value::HighPassChebyshev(_) | Value::BandPassChebyshev(_) | Value::Saturate(_)
            | Value::Compress(_) | Value::Delay(_) | Value::Echo(_) | Value::SampleAndHold(_)
            | Value::Quantize(_) | Value::DownSample(_) | Value::QuantizeToScale(_)
            | Value::Reverb(_) | Value::EnvelopeFollower(_) | Value::NoiseGate(_)
            | Value::ClockedMidiNoteMonophonicLooper(_)
            | Value::Oscillator(_) => Err(()),
            Value::Gate(v) => Ok(v.into()),
            Value::GateU8(_) => Err(()),
            Value::Sample(_) => Err(()),
            Value::Su32(_) => Err(())
        }
    }
}

impl TryFrom<Value> for Su8 {
    type Error = ();

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::Su8(v) => Ok(v.into()),
            _ => Err(())
        }
    }
}

impl TryFrom<Value> for Sample {
    type Error = ();

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::Sample(v) => Ok(v.into()),
            _ => Err(())
        }
    }
}

impl TryFrom<Value> for Trigger {
    type Error = ();

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::Terminal(_) | Value::Sfreq(_) | Value::Sf64(_) | Value::Sf32(_)  
            | Value::Sbool(_) | Value::Su8(_) | Value::Gate(_) | Value::Zip(_)
            | Value::LowPassButterworth(_) | Value::HighPassButterworth(_)
            | Value::LowPassMoogLadder(_) | Value::SWaveform(_)
            | Value::BandPassButterworth(_) | Value::LowPassChebyshev(_)
            | Value::HighPassChebyshev(_) | Value::BandPassChebyshev(_) | Value::Saturate(_)
            | Value::Compress(_) | Value::Delay(_) | Value::Echo(_) | Value::SampleAndHold(_)
            | Value::Quantize(_) | Value::DownSample(_) | Value::QuantizeToScale(_)
            | Value::Reverb(_) | Value::EnvelopeFollower(_) | Value::NoiseGate(_)
            | Value::ClockedMidiNoteMonophonicLooper(_)
            | Value::Oscillator(_) => Err(()),
            Value::Trigger(v) => Ok(v.into()),
            Value::GateU8(_) => Err(()),
            Value::Sample(_) => Err(()),
            Value::Su32(_) => Err(())
        }
    }
}

impl TryFrom<Value> for Signal<Waveform> {
    type Error = ();

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::Terminal(_) | Value::Sfreq(_) | Value::Sf64(_) | Value::Sf32(_) 
            | Value::Sbool(_) | Value::Su8(_) | Value::Gate(_) | Value::Trigger(_)
            | Value::Zip(_) | Value::LowPassButterworth(_) | Value::HighPassButterworth(_)
            | Value::LowPassMoogLadder(_)
            | Value::BandPassButterworth(_) | Value::LowPassChebyshev(_)
            | Value::HighPassChebyshev(_) | Value::BandPassChebyshev(_) | Value::Saturate(_)
            | Value::Compress(_) | Value::Delay(_) | Value::Echo(_) | Value::SampleAndHold(_)
            | Value::Quantize(_) | Value::DownSample(_) | Value::QuantizeToScale(_)
            | Value::Reverb(_) | Value::EnvelopeFollower(_) | Value::NoiseGate(_)
            | Value::ClockedMidiNoteMonophonicLooper(_)
            | Value::Oscillator(_) => Err(()),
            Value::SWaveform(v) => Ok(v.into()),
            Value::GateU8(_) => Err(()),
            Value::Sample(_) => Err(()),
            Value::Su32(_) => Err(())
        }
    }
}

impl TryFrom<Value> for Sfreq {
    type Error = ();

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::Terminal(_) | Value::Sbool(_)
            | Value::Su8(_) | Value::Gate(_) | Value::Trigger(_) | Value::Zip(_)
            | Value::LowPassButterworth(_) | Value::HighPassButterworth(_)
            | Value::LowPassMoogLadder(_)
            | Value::SWaveform(_) | Value::BandPassButterworth(_) | Value::LowPassChebyshev(_)
            | Value::HighPassChebyshev(_) | Value::BandPassChebyshev(_) | Value::Saturate(_)
            | Value::Compress(_) | Value::Delay(_) | Value::Echo(_) | Value::SampleAndHold(_)
            | Value::Quantize(_) | Value::DownSample(_) | Value::QuantizeToScale(_)
            | Value::Reverb(_) | Value::EnvelopeFollower(_) | Value::NoiseGate(_)
            | Value::ClockedMidiNoteMonophonicLooper(_)
            | Value::Oscillator(_) => Err(()),
            Value::Sf64(v) => Ok(v.map(|v| Freq::from_hz(v))),
            Value::Sf32(v) => Ok(v.map(|v| Freq::from_hz(v as f64))),
            Value::Sfreq(v) => Ok(v.into()),
            Value::GateU8(_) => Err(()),
            Value::Sample(_) => Err(()),
            Value::Su32(_) => Err(())
        }
    }
}

impl TryFrom<Value> for Sbool {
    type Error = ();

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::Terminal(Terminal::Bool(value)) => Ok(const_(value)),
            Value::Sbool(signal) => Ok(signal),
            _ => Err(())
        }
    }
}

impl TryFrom<Value> for Su32 {
    type Error = ();

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::Terminal(Terminal::U32(value)) => Ok(const_(value)),
            _ => Err(())
        }
    }
}

impl TryFrom<Terminal> for Sf64 {
    type Error = ();

    fn try_from(value: Terminal) -> Result<Self, Self::Error> {
        match value {
            Terminal::F64(v) => Ok(v.into()),
            Terminal::F32(v) => Ok(v.into()),
            Terminal::U8(v) => Err(()),
            Terminal::Bool(v) => Err(()),
            Terminal::Freq(v) => Err(()),
            Terminal::Waveform(v) => Err(()),
            Terminal::U32(v) => Err(())
        }
    }
}


impl From<f64> for Expr {
    fn from(value: f64) -> Self {
        Expr::Constant(Terminal::F64(value))
    }
}