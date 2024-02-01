use crate::{
    music::Note,
    signal::{Gate, Sf64, Signal, SignalCtx, Trigger},
};
use std::{cell::RefCell, mem, rc::Rc};

#[derive(Clone)]
pub struct VoiceDesc {
    pub note: Signal<Note>,
    pub key_down: Gate,
    pub key_press: Trigger,
    pub velocity_01: Sf64,
}

#[derive(Clone, Copy, Debug)]
pub struct KeyEvent {
    pub note: Note,
    pub pressed: bool,
    pub velocity_01: f64,
}

#[derive(Clone, Copy, Default)]
struct HeldKey {
    note: Note,
    velocity_01: f64,
}

impl HeldKey {
    fn from_key_event(key_event: &KeyEvent) -> Self {
        Self {
            note: key_event.note,
            velocity_01: key_event.velocity_01,
        }
    }
}

#[derive(Default)]
struct PolyphonicVoice {
    key: HeldKey,
    /// Key information is retained until the voice is reused to let the envelope play out
    /// after a key is released.
    key_down: bool,
    key_press: bool,
    key_press_sample_index: u64,
}

/// Choose the next voice to use to play a note. There may be multiple implementations of this
/// because there's no one way to choose which voice to use to play a note when all available
/// voices are currently playing notes.
trait PolyphonicVoiceReusePolicy {
    fn choose_voice_index(&mut self, voices: &[PolyphonicVoice]) -> Option<usize>;
}

mod polyphonic_voice_reuse_policy {
    use super::{PolyphonicVoice, PolyphonicVoiceReusePolicy};
    use std::collections::BinaryHeap;

    #[derive(PartialEq, Eq)]
    struct GenerationalEntry {
        index: usize,
        sample_index: u64,
    }

    impl PartialOrd for GenerationalEntry {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.sample_index.cmp(&other.sample_index).reverse())
        }
    }

    impl Ord for GenerationalEntry {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.sample_index.cmp(&other.sample_index).reverse()
        }
    }

    /// If all voices are in use, reuse the Nth oldest where N is a configurable parameter. This is
    /// intended to capture the benefits of Fifo and Lifo. If N=3 then it will be possible to hold
    /// down 3 notes indefinitely, and a Fifo policy will be used for the remainder. If there are
    /// available voices, the oldest one will be chosen to give envelopes as much time as possible
    /// to play their releases. It's generational in the sense that held keys are divided into the
    /// n oldest keys and remaining younger keys.
    pub struct Generational {
        n: usize,
        heap: BinaryHeap<GenerationalEntry>,
    }

    impl Generational {
        pub fn new(n: usize) -> Self {
            Self {
                n,
                heap: BinaryHeap::new(),
            }
        }
    }

    impl PolyphonicVoiceReusePolicy for Generational {
        fn choose_voice_index(&mut self, voices: &[PolyphonicVoice]) -> Option<usize> {
            self.heap.clear();
            let mut oldest_available_voice: Option<GenerationalEntry> = None;
            let mut nth_oldest_unavailable_voice = None;
            for (i, voice) in voices.iter().enumerate() {
                if voice.key_down {
                    self.heap.push(GenerationalEntry {
                        index: i,
                        sample_index: voice.key_press_sample_index,
                    });
                    if self.heap.len() > self.n {
                        nth_oldest_unavailable_voice = self.heap.pop();
                    }
                } else {
                    if let Some(ref entry) = oldest_available_voice {
                        if voice.key_press_sample_index < entry.sample_index {
                            oldest_available_voice = Some(GenerationalEntry {
                                index: i,
                                sample_index: voice.key_press_sample_index,
                            });
                        }
                    } else {
                        oldest_available_voice = Some(GenerationalEntry {
                            index: i,
                            sample_index: voice.key_press_sample_index,
                        });
                    }
                }
            }
            // Favour available voices but fall back to unavailable ones if necessary.
            oldest_available_voice
                .or(nth_oldest_unavailable_voice)
                .map(|entry| entry.index)
        }
    }
}

impl VoiceDesc {
    fn monophonic_from_key_events(key_events: Signal<Vec<KeyEvent>>) -> Self {
        #[derive(Default)]
        struct State {
            held_keys: Vec<HeldKey>,
            /// The information to use when no key is held
            sticky: HeldKey,
            key_just_pressed: bool,
        }

        impl State {
            fn handle_key_event(&mut self, key_event: &KeyEvent) {
                // Remove the held key if it already exists. This assumes there are no duplicate
                // keys in the vector.
                for (i, held_key) in self.held_keys.iter().enumerate() {
                    if held_key.note == key_event.note {
                        self.held_keys.remove(i);
                        break;
                    }
                }
                self.sticky = HeldKey::from_key_event(key_event);
                if key_event.pressed {
                    self.held_keys.push(self.sticky);
                    self.key_just_pressed = true;
                }
            }
            fn last(&self) -> Option<&HeldKey> {
                self.held_keys.last()
            }
        }

        let state = Rc::new(RefCell::new(State::default()));
        let update_state = key_events.map({
            let state = Rc::clone(&state);
            move |key_events_this_tick| {
                let mut state = state.borrow_mut();
                for key_event in &key_events_this_tick {
                    state.handle_key_event(key_event);
                }
            }
        });
        let note = update_state.then({
            let state = Rc::clone(&state);
            move || {
                let state = state.borrow();
                if let Some(last) = state.last() {
                    last.note
                } else {
                    state.sticky.note
                }
            }
        });
        let key_down = update_state
            .then({
                let state = Rc::clone(&state);
                move || state.borrow().last().is_some()
            })
            .to_gate();
        let key_press = update_state
            .then({
                let state = Rc::clone(&state);
                move || {
                    let mut state = state.borrow_mut();
                    mem::replace(&mut state.key_just_pressed, false)
                }
            })
            .to_trigger_raw();
        let velocity_01 = update_state.then({
            let state = Rc::clone(&state);
            move || {
                let state = state.borrow();
                if let Some(last) = state.last() {
                    last.velocity_01
                } else {
                    state.sticky.velocity_01
                }
            }
        });
        Self {
            note,
            key_down,
            key_press,
            velocity_01,
        }
    }

    fn polyphonic_from_key_events<P: PolyphonicVoiceReusePolicy + 'static>(
        reuse_policy: P,
        num_voices: usize,
        key_events: Signal<Vec<KeyEvent>>,
    ) -> Vec<Self> {
        struct State<P: PolyphonicVoiceReusePolicy> {
            reuse_policy: P,
            voices: Vec<PolyphonicVoice>,
        }

        impl<P: PolyphonicVoiceReusePolicy> State<P> {
            fn new(reuse_policy: P, num_voices: usize) -> Self {
                Self {
                    reuse_policy,
                    voices: (0..num_voices).map(|_| Default::default()).collect(),
                }
            }

            fn handle_key_event(&mut self, key_event: &KeyEvent, ctx: &SignalCtx) {
                if key_event.pressed {
                    if let Some(i) = self.reuse_policy.choose_voice_index(&self.voices) {
                        let voice = &mut self.voices[i];
                        *voice = PolyphonicVoice {
                            key: HeldKey::from_key_event(key_event),
                            key_down: true,
                            key_press: true,
                            key_press_sample_index: ctx.sample_index,
                        };
                    }
                } else {
                    for voice in &mut self.voices {
                        if voice.key.note == key_event.note {
                            voice.key_down = false;
                            voice.key.velocity_01 = key_event.velocity_01;
                        }
                    }
                }
            }
        }

        let state = Rc::new(RefCell::new(State::new(reuse_policy, num_voices)));
        let update_state = key_events.map_ctx({
            let state = Rc::clone(&state);
            move |key_events_this_tick, ctx| {
                let mut state = state.borrow_mut();
                for key_event in &key_events_this_tick {
                    state.handle_key_event(key_event, ctx);
                }
            }
        });
        (0..num_voices)
            .map(|i| {
                let note = update_state.then({
                    let state = Rc::clone(&state);
                    move || {
                        let state = state.borrow();
                        state.voices[i].key.note
                    }
                });
                let key_down = update_state
                    .then({
                        let state = Rc::clone(&state);
                        move || {
                            let state = state.borrow();
                            state.voices[i].key_down
                        }
                    })
                    .to_gate();
                let key_press = update_state
                    .then({
                        let state = Rc::clone(&state);
                        move || {
                            let mut state = state.borrow_mut();
                            mem::replace(&mut state.voices[i].key_press, false)
                        }
                    })
                    .to_trigger_raw();
                let velocity_01 = update_state.then({
                    let state = Rc::clone(&state);
                    move || {
                        let state = state.borrow();
                        state.voices[i].key.velocity_01
                    }
                });
                Self {
                    note,
                    key_down,
                    key_press,
                    velocity_01,
                }
            })
            .collect()
    }
}

impl Signal<Vec<KeyEvent>> {
    pub fn voice_desc_monophonic(self) -> VoiceDesc {
        VoiceDesc::monophonic_from_key_events(self)
    }

    pub fn voice_descs_polyphonic_generational(
        self,
        num_persistent_voices: usize,
        num_transient_voices: usize,
    ) -> Vec<VoiceDesc> {
        let reuse_policy = polyphonic_voice_reuse_policy::Generational::new(num_persistent_voices);
        VoiceDesc::polyphonic_from_key_events(
            reuse_policy,
            num_persistent_voices + num_transient_voices,
            self,
        )
    }
}
