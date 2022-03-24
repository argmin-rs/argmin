// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

#[cfg(feature = "serde1")]
mod file;

#[cfg(feature = "serde1")]
pub use crate::core::checkpointing::file::FileCheckpoint;

use crate::core::Error;
use std::default::Default;
use std::fmt::Display;

/// An interface for checkpointing methods
///
/// Handles saving of a checkpoint. The methods [`save`](`Checkpoint::save`) (saving the
/// checkpoint), [`load`](`Checkpoint::load`) (loading a checkpoint) are mandatory to implement.
/// The method [`save_cond`](`Checkpoint::save_cond`) determines if the conditions for calling
/// `save` are met, and if yes, calles `save`. [`freqency`](`Checkpoint::frequency`) returns the
/// conditions in form of a [`CheckpointingFrequency`].
///
/// # Example
///
/// ```
/// # extern crate serde;
/// use argmin::core::Error;
/// use argmin::core::checkpointing::{Checkpoint, CheckpointingFrequency};
/// use serde::{Serialize, de::DeserializeOwned};
///
/// struct MyCheckpoint {
///    frequency: CheckpointingFrequency,
///    // ..
/// }
///
/// impl<S, I> Checkpoint<S, I> for MyCheckpoint
/// where
///     // Both `solver` (`S`) and `state` (`I`) (probably) need to be (de)serializable
///     S: Serialize + DeserializeOwned,
///     I: Serialize + DeserializeOwned,
/// #     S: Default,
/// #     I: Default,
/// {
///     fn save(&self, solver: &S, state: &I) -> Result<(), Error> {
///         // Save `solver` and `state`
///         Ok(())
///     }
///
///     fn load(&self) -> Result<Option<(S, I)>, Error> {
///         // Load `solver` and `state` from checkpoint
///         // Return `Ok(None)` in case checkpoint is not found.
/// #         let solver = S::default();
/// #         let state = I::default();
///         Ok(Some((solver, state)))
///     }
///
///     fn frequency(&self) -> CheckpointingFrequency {
///         self.frequency
///     }
/// }
/// # fn main() {}
/// ```
pub trait Checkpoint<S, I> {
    /// Save a checkpoint
    ///
    /// Gets a reference to the current `solver` of type `S` and to the current `state` of type
    /// `I`. Both solver and state can maintain state. Optimization problems itself are not allowed
    /// to have state which changes during an optimization (at least not in the context of
    /// checkpointing).
    fn save(&self, solver: &S, state: &I) -> Result<(), Error>;

    /// Saves a checkpoint when the checkpointing condition is met.
    ///
    /// Calls [`save`](`Checkpoint::save`) in each iteration (`CheckpointingFrequency::Always`),
    /// every X iterations (`CheckpointingFrequency::Every(X)`) or never
    /// (`CheckpointingFrequency::Never`).
    fn save_cond(&self, solver: &S, state: &I, iter: u64) -> Result<(), Error> {
        match self.frequency() {
            CheckpointingFrequency::Always => self.save(solver, state)?,
            CheckpointingFrequency::Every(it) if iter % it == 0 => self.save(solver, state)?,
            CheckpointingFrequency::Never | CheckpointingFrequency::Every(_) => {}
        };
        Ok(())
    }

    /// Loads a saved checkpoint
    ///
    /// Returns the solver of type `S` and the `state` of type `I`.
    fn load(&self) -> Result<Option<(S, I)>, Error>;

    /// Indicates how often checkpoints should be saved
    ///
    /// Returns enum `CheckpointingFrequency`.
    fn frequency(&self) -> CheckpointingFrequency;
}

/// Defines at which intervals a checkpoint is saved.
///
/// # Example
///
/// ```
/// use argmin::core::checkpointing::CheckpointingFrequency;
///
/// // A checkpoint every 10 iterations
/// let every_10 = CheckpointingFrequency::Every(10);
///
/// // A checkpoint in each iteration
/// let always = CheckpointingFrequency::Always;
///
/// // The default is `CheckpointingFrequency::Never`
/// assert_eq!(CheckpointingFrequency::default(), CheckpointingFrequency::Never);
/// ```
#[derive(Clone, Eq, PartialEq, Debug, Hash, Copy)]
pub enum CheckpointingFrequency {
    /// Never create checkpoint
    Never,
    /// Create checkpoint every N iterations
    Every(u64),
    /// Create checkpoint in every iteration
    Always,
}

impl Default for CheckpointingFrequency {
    fn default() -> CheckpointingFrequency {
        CheckpointingFrequency::Never
    }
}

impl Display for CheckpointingFrequency {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            CheckpointingFrequency::Never => write!(f, "Never"),
            CheckpointingFrequency::Every(i) => write!(f, "Every({})", i),
            CheckpointingFrequency::Always => write!(f, "Always"),
        }
    }
}
