// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Observers
//!
//! Observers are called after an iteration of a solver was performed and enable the user to observe
//! the current state of the optimization. This can be used for logging or writing the current
//! parameter vector to disk.

pub mod file;
pub mod slog_logger;
#[cfg(feature = "visualizer")]
pub mod visualizer;

use crate::core::{ArgminFloat, ArgminKV, ArgminOp, Error, IterState};
use serde::{Deserialize, Serialize};
use std::default::Default;
use std::sync::{Arc, Mutex};

pub use file::*;
pub use slog_logger::*;
#[cfg(feature = "visualizer")]
pub use visualizer::*;

/// Defines the interface every Observer needs to expose
pub trait Observe<O: ArgminOp, F> {
    /// Called once at the beginning of the execution of the solver.
    ///
    /// Parameters:
    ///
    /// `name`: Name of the solver
    /// `kv`: Key-Value storage of initial configurations defined by the `Solver`
    fn observe_init(&self, _name: &str, _kv: &ArgminKV) -> Result<(), Error> {
        Ok(())
    }

    /// Called at every iteration of the solver
    ///
    /// Parameters
    ///
    /// `state`: Current state of the solver. See documentation of `IterState` for details.
    /// `kv`: Key-Value store of relevant variables defined by the `Solver`
    fn observe_iter(&mut self, _state: &IterState<O, F>, _kv: &ArgminKV) -> Result<(), Error> {
        Ok(())
    }
}

/// Container for observers which acts just like a single `Observe`r by implementing `Observe` on
/// it.
#[derive(Clone, Default)]
pub struct Observer<O, F> {
    /// Vector of `Observe`rs with the corresponding `ObserverMode`
    observers: Vec<(Arc<Mutex<dyn Observe<O, F>>>, ObserverMode)>,
}

impl<O: ArgminOp, F> Observer<O, F> {
    /// Constructor
    pub fn new() -> Self {
        Observer { observers: vec![] }
    }

    /// Push another `Observe` to the `observer` field
    pub fn push<OBS: Observe<O, F> + 'static>(
        &mut self,
        observer: OBS,
        mode: ObserverMode,
    ) -> &mut Self {
        self.observers.push((Arc::new(Mutex::new(observer)), mode));
        self
    }
}

/// By implementing `Observe` for `Observer` we basically allow a set of `Observer`s to be used
/// just like a single `Observe`r.
impl<O: ArgminOp, F: ArgminFloat> Observe<O, F> for Observer<O, F> {
    /// Initial observation
    /// This is called after the initialization in an `Executor` and gets the name of the solver as
    /// string and a `ArgminKV` which includes some solver-specific information.
    fn observe_init(&self, msg: &str, kv: &ArgminKV) -> Result<(), Error> {
        for l in self.observers.iter() {
            l.0.lock().unwrap().observe_init(msg, kv)?
        }
        Ok(())
    }

    /// This is called after every iteration and gets the current `state` of the solver as well as
    /// a `KV` which can include solver-specific information
    /// This respects the `ObserverMode`: Every `Observe`r is only called as often as specified.
    fn observe_iter(&mut self, state: &IterState<O, F>, kv: &ArgminKV) -> Result<(), Error> {
        use ObserverMode::*;
        for l in self.observers.iter_mut() {
            let iter = state.get_iter();
            let observer = &mut l.0.lock().unwrap();
            match l.1 {
                Always => observer.observe_iter(state, kv),
                Every(i) if iter % i == 0 => observer.observe_iter(state, kv),
                NewBest if state.is_best() => observer.observe_iter(state, kv),
                Never | Every(_) | NewBest => Ok(()),
            }?
        }
        Ok(())
    }
}

/// This is used to indicate how often the observer will observe the status. `Never` deactivates
/// the observer, `Always` and `Every(i)` will call the observer in every or every ith iteration,
/// respectively. `NewBest` will call the observer only, if a new best solution is found.
#[derive(Copy, Clone, Serialize, Deserialize, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum ObserverMode {
    /// Never call the observer
    Never,
    /// Call observer in every iteration
    Always,
    /// Call observer every N iterations
    Every(u64),
    /// Call observer when new best is found
    NewBest,
}

impl Default for ObserverMode {
    /// The default is `Always`
    fn default() -> ObserverMode {
        ObserverMode::Always
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::core::MinimalNoOperator;
//
//     send_sync_test!(observer, Observer<MinimalNoOperator>);
//     send_sync_test!(observermode, ObserverMode);
// }
