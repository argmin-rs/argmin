// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Checkpointing
//!
//! Checkpointing is a useful mechanism for mitigating the effects of crashes when software is run
//! in an unstable environment, particularly for long run times. Checkpoints are saved regularly
//! with a user-chosen frequency. Optimizations can then be resumed from a given checkpoint after a
//! crash.
//!
//! For saving checkpoints to disk, `FileCheckpoint` is provided in the `argmin-checkpointing-file`
//! crate.
//! Via the `Checkpoint` trait other checkpointing approaches can be implemented.
//!
//! The `CheckpointingFrequency` defines how often checkpoints are saved and can be chosen to be
//! either `Always` (every iteration), `Every(u64)` (every Nth iteration) or `Never`.
//!
//! The following example shows how the `checkpointing` method is used to activate checkpointing.
//! If no checkpoint is available on disk, an optimization will be started from scratch. If the run
//! crashes and a checkpoint is found on disk, then it will resume from the checkpoint.
//!
//! ## Example
//!
//! ```rust
//! # extern crate argmin;
//! # extern crate argmin_testfunctions;
//! # use argmin::core::{CostFunction, Error, Executor, Gradient, observers::ObserverMode};
//! # #[cfg(feature = "serde1")]
//! use argmin::core::checkpointing::CheckpointingFrequency;
//! # #[cfg(feature = "serde1")]
//! use argmin_checkpointing_file::FileCheckpoint;
//! # use argmin_observer_slog::SlogLogger;
//! # use argmin::solver::landweber::Landweber;
//! # use argmin_testfunctions::{rosenbrock, rosenbrock_derivative};
//! #
//! # #[derive(Default)]
//! # struct Rosenbrock {}
//! #
//! # /// Implement `CostFunction` for `Rosenbrock`
//! # impl CostFunction for Rosenbrock {
//! #     /// Type of the parameter vector
//! #     type Param = Vec<f64>;
//! #     /// Type of the return value computed by the cost function
//! #     type Output = f64;
//! #
//! #     /// Apply the cost function to a parameter `p`
//! #     fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
//! #         Ok(rosenbrock(p))
//! #     }
//! # }
//! #
//! # /// Implement `Gradient` for `Rosenbrock`
//! # impl Gradient for Rosenbrock {
//! #     /// Type of the parameter vector
//! #     type Param = Vec<f64>;
//! #     /// Type of the return value computed by the cost function
//! #     type Gradient = Vec<f64>;
//! #
//! #     /// Compute the gradient at parameter `p`.
//! #     fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
//! #         Ok(rosenbrock_derivative(p))
//! #     }
//! # }
//! #
//! # fn run() -> Result<(), Error> {
//! #     // define initial parameter vector
//! #     let init_param: Vec<f64> = vec![1.2, 1.2];
//! #     let my_optimization_problem = Rosenbrock {};
//! #
//! #     let iters = 35;
//! #     let solver = Landweber::new(0.001);
//!
//! // [...]
//!
//! # #[cfg(feature = "serde1")]
//! let checkpoint = FileCheckpoint::new(
//!     ".checkpoints",
//!     "optim",
//!     CheckpointingFrequency::Every(20)
//! );
//!
//! #
//! # #[cfg(feature = "serde1")]
//! let res = Executor::new(my_optimization_problem, solver)
//!     .configure(|config| config.param(init_param).max_iters(iters))
//!     .checkpointing(checkpoint)
//!     .run()?;
//!
//! // [...]
//! #
//! #     Ok(())
//! # }
//! #
//! # fn main() {
//! #     if let Err(ref e) = run() {
//! #         println!("{}", e);
//! #     }
//! # }
//! ```

use crate::core::Error;
use std::default::Default;
use std::fmt::Display;

/// An interface for checkpointing methods
///
/// Handles saving of a checkpoint. The methods [`save`](`Checkpoint::save`) (saving the
/// checkpoint), [`load`](`Checkpoint::load`) (loading a checkpoint) are mandatory to implement.
/// The method [`save_cond`](`Checkpoint::save_cond`) determines if the conditions for calling
/// `save` are met, and if yes, calls `save`. [`frequency`](`Checkpoint::frequency`) returns the
/// conditions in form of a [`CheckpointingFrequency`].
///
/// # Example
///
/// ```
/// use argmin::core::Error;
/// use argmin::core::checkpointing::{Checkpoint, CheckpointingFrequency};
/// # #[cfg(feature = "serde1")]
/// use serde::{Serialize, de::DeserializeOwned};
///
/// struct MyCheckpoint {
///    frequency: CheckpointingFrequency,
///    // ..
/// }
///
/// # #[cfg(feature = "serde1")]
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
/// // The default is `CheckpointingFrequency::Always`
/// assert_eq!(CheckpointingFrequency::default(), CheckpointingFrequency::Always);
/// ```
#[derive(Clone, Eq, PartialEq, Debug, Hash, Copy, Default)]
pub enum CheckpointingFrequency {
    /// Never create checkpoint
    Never,
    /// Create checkpoint every N iterations
    Every(u64),
    /// Create checkpoint in every iteration
    #[default]
    Always,
}

impl Display for CheckpointingFrequency {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            CheckpointingFrequency::Never => write!(f, "Never"),
            CheckpointingFrequency::Every(i) => write!(f, "Every({i})"),
            CheckpointingFrequency::Always => write!(f, "Always"),
        }
    }
}
