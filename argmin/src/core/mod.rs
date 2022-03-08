// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Argmin Optimizaton toolbox core
//!
//! This module contains the traits and types necessary for implementing optimization algorithms
//! and tools for observing the state of optimization runs and checkpointing.

// I really do not like the a..=b syntax
#![allow(clippy::range_plus_one)]

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
/// iteration state
mod state;
/// Definition of termination reasons
mod termination;

pub use anyhow::Error;
#[cfg(feature = "serde1")]
pub use checkpointing::{load_checkpoint, Checkpoint, CheckpointMode};
pub use errors::ArgminError;
pub use executor::Executor;
pub use float::ArgminFloat;
pub use kv::KV;
pub use observers::*;
pub use opwrapper::OpWrapper;
pub use problem::{CostFunction, Gradient, Hessian, Jacobian, LinearProgram, Modify, Operator};
#[cfg(test)]
pub use pseudooperator::PseudoOperator;
pub use result::OptimizationResult;
pub use serialization::{DeserializeOwnedAlias, SerializeAlias};
pub use state::{IterState, LinearProgramState, State};
pub use termination::TerminationReason;

/// Solver
///
/// Every solver needs to implement this trait.
pub trait Solver<O, I: State>: SerializeAlias {
    /// Name of the solver
    const NAME: &'static str = "UNDEFINED";

    /// Computes one iteration of the algorithm.
    fn next_iter(&mut self, op: &mut OpWrapper<O>, state: I) -> Result<(I, Option<KV>), Error>;

    /// Initializes the algorithm
    ///
    /// This is executed before any iterations are performed. It can be used to perform
    /// precomputations. The default implementation corresponds to doing nothing.
    fn init(&mut self, _op: &mut OpWrapper<O>, state: I) -> Result<(I, Option<KV>), Error> {
        Ok((state, Some(KV::new())))
    }

    /// Checks whether basic termination reasons apply.
    ///
    /// Terminate if
    ///
    /// 1) algorithm was terminated somewhere else in the Executor
    /// 2) iteration count exceeds maximum number of iterations
    /// 3) cost is lower than target cost
    ///
    /// This can be overwritten in a `Solver` implementation; however it is not advised.
    fn terminate_internal(&mut self, state: &I) -> TerminationReason {
        let solver_terminate = self.terminate(state);
        if solver_terminate.terminated() {
            return solver_terminate;
        }
        if state.get_iter() >= state.get_max_iters() {
            return TerminationReason::MaxItersReached;
        }
        if state.get_cost() <= state.get_target_cost() {
            return TerminationReason::TargetCostReached;
        }
        TerminationReason::NotTerminated
    }

    /// Checks whether the algorithm must be terminated
    fn terminate(&mut self, _state: &I) -> TerminationReason {
        TerminationReason::NotTerminated
    }
}

/// Defines a common interface for line search methods.
pub trait LineSearch<P, F> {
    /// Set the search direction
    fn set_search_direction(&mut self, direction: P);

    /// Set the initial step length
    fn set_init_alpha(&mut self, step_length: F) -> Result<(), Error>;
}

/// Defines a common interface to methods which calculate approximate steps for trust region
/// methods.
pub trait TrustRegion<F> {
    /// Set the initial step length
    fn set_radius(&mut self, radius: F);
}
//
/// Common interface for beta update methods (Nonlinear-CG)
pub trait NLCGBetaUpdate<G, P, F: ArgminFloat>: SerializeAlias {
    /// Update beta
    /// Parameter 1: \nabla f_k
    /// Parameter 2: \nabla f_{k+1}
    /// Parameter 3: p_k
    fn update(&self, nabla_f_k: &G, nabla_f_k_p_1: &G, p_k: &P) -> F;
}
