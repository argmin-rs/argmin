// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Argmin Optimization toolbox core
//!
//! This module contains the traits and types necessary for implementing optimization algorithms
//! and tools for observing the state of optimization runs and checkpointing.

/// Macros
#[macro_use]
pub mod macros;
/// Checkpointing
#[cfg(feature = "serde1")]
mod checkpointing;
/// Error handling
mod errors;
/// Executor
pub mod executor;
/// Trait alias for float types
mod float;
/// Key value datastructure
mod kv;
/// Observers;
mod observers;
/// Wrapper around operators which keeps track of function evaluation counts
mod opwrapper;
/// Traits needed to define optimization problems
mod problem;
/// Pseudo Operator
mod pseudooperator;
/// Definition of the return type of the solvers
mod result;
/// Trait alias for `serde`s `Serialize` and `DeserializeOwned`
mod serialization;
/// `Solver` trait
mod solver;
/// iteration state
mod state;
/// Definition of termination reasons
mod termination;

pub use crate::solver::conjugategradient::NLCGBetaUpdate;
pub use crate::solver::linesearch::LineSearch;
pub use crate::solver::trustregion::TrustRegionRadius;
pub use anyhow::Error;
#[cfg(feature = "serde1")]
pub use checkpointing::{load_checkpoint, Checkpoint, CheckpointMode};
pub use errors::ArgminError;
pub use executor::Executor;
pub use float::ArgminFloat;
pub use kv::KV;
pub use observers::*;
pub use opwrapper::OpWrapper;
pub use problem::{CostFunction, Gradient, Hessian, Jacobian, LinearProgram, Operator};
#[cfg(test)]
pub use pseudooperator::PseudoOperator;
pub use result::OptimizationResult;
pub use serialization::{DeserializeOwnedAlias, SerializeAlias};
pub use solver::Solver;
pub use state::{IterState, LinearProgramState, State};
pub use termination::TerminationReason;
