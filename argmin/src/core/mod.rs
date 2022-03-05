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
/// Error handling
mod errors;
/// Executor
pub mod executor;
/// Key value datastructure
mod kv;
/// Phony Operator
// #[cfg(test)]
mod nooperator;
/// Observers;
mod observers;
/// Wrapper around operators which keeps track of function evaluation counts
mod opwrapper;
/// Definition of the return type of the solvers
mod result;
/// Serialization of `ArgminSolver`s
#[cfg(feature = "serde1")]
mod serialization;
/// iteration state
mod state;
/// Definition of termination reasons
mod termination;

pub use anyhow::Error;
pub use errors::ArgminError;
pub use executor::Executor;
// pub use iterstate::{IterState, LinearProgramState, State};
pub use kv::KV;
pub use nooperator::{MinimalNoOperator, NoOperator};
use num_traits::{Float, FloatConst, FromPrimitive, ToPrimitive};
pub use observers::*;
pub use opwrapper::OpWrapper;
pub use result::OptimizationResult;
#[cfg(feature = "serde1")]
use serde::{de::DeserializeOwned, Serialize};
#[cfg(feature = "serde1")]
pub use serialization::{load_checkpoint, ArgminCheckpoint, CheckpointMode};
pub use state::{IterState, LinearProgramState, State};
use std::fmt::{Debug, Display};
pub use termination::TerminationReason;

/// Trait alias to simplify common trait bounds
pub trait ArgminFloat:
    Float
    + FloatConst
    + FromPrimitive
    + ToPrimitive
    + Debug
    + Display
    + SerializeAlias
    + DeserializeOwnedAlias
{
}
impl<I> ArgminFloat for I where
    I: Float
        + FloatConst
        + FromPrimitive
        + ToPrimitive
        + Debug
        + Display
        + SerializeAlias
        + DeserializeOwnedAlias
{
}

/// TODO
pub trait Operator {
    /// Type of the parameter vector
    type Param: Clone + SerializeAlias + DeserializeOwnedAlias;
    /// Output of the operator
    type Output: Clone + SerializeAlias + DeserializeOwnedAlias;

    /// Applies the operator to parameters
    fn apply(&self, param: &Self::Param) -> Result<Self::Output, Error>;
}

/// TODO
pub trait CostFunction {
    /// Type of the parameter vector
    type Param: Clone + SerializeAlias + DeserializeOwnedAlias;
    /// Output of the cost function
    type Output: Clone + SerializeAlias + DeserializeOwnedAlias;

    /// Compute cost function
    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error>;
}

/// TODO
pub trait Gradient {
    /// Type of the parameter vector
    type Param: Clone + SerializeAlias + DeserializeOwnedAlias;
    /// Type of the gradient
    type Gradient: Clone + SerializeAlias + DeserializeOwnedAlias;

    /// Compute gradient
    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, Error>;
}

/// TODO
pub trait Hessian {
    /// Type of the parameter vector
    type Param: Clone + SerializeAlias + DeserializeOwnedAlias;
    /// Type of the Hessian
    type Hessian: Clone + SerializeAlias + DeserializeOwnedAlias;

    /// Compute Hessian
    fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, Error>;
}

/// TODO
pub trait Jacobian {
    /// Type of the parameter vector
    type Param: Clone + SerializeAlias + DeserializeOwnedAlias;
    /// Output of the cost function
    type Jacobian: Clone + SerializeAlias + DeserializeOwnedAlias;

    /// Compute Jacobian
    fn jacobian(&self, param: &Self::Param) -> Result<Self::Jacobian, Error>;
}

/// TODO
pub trait Modify {
    /// Type of the parameter vector
    type Param: Clone + SerializeAlias + DeserializeOwnedAlias;
    /// Output TODO
    type Output: Clone + SerializeAlias + DeserializeOwnedAlias;
    /// Precision of floats
    type Float;

    /// Compute Jacobian
    fn modify(&self, param: &Self::Param, _extent: Self::Float) -> Result<Self::Output, Error>;
}

/// Problems which implement this trait can be used for linear programming solvers
pub trait LinearProgram {
    /// Type of the parameter vector
    type Param: Clone + SerializeAlias + DeserializeOwnedAlias;
    /// Precision of floats
    type Float: ArgminFloat;

    /// TODO c for linear programs
    /// Those three could maybe be merged into a single function; name unclear
    fn c(&self) -> Result<Vec<Self::Float>, Error> {
        Err(ArgminError::NotImplemented {
            text: "Method `c` of ArgminOp trait not implemented!".to_string(),
        }
        .into())
    }

    /// TODO b for linear programs
    fn b(&self) -> Result<Vec<Self::Float>, Error> {
        Err(ArgminError::NotImplemented {
            text: "Method `b` of ArgminOp trait not implemented!".to_string(),
        }
        .into())
    }

    /// TODO A for linear programs
    #[allow(non_snake_case)]
    fn A(&self) -> Result<Vec<Vec<Self::Float>>, Error> {
        Err(ArgminError::NotImplemented {
            text: "Method `A` of ArgminOp trait not implemented!".to_string(),
        }
        .into())
    }
}

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

/// If the `serde1` feature is set, it acts as an alias for `Serialize` and is implemented for all
/// types which implement `Serialize`. If `serde1` is not set, it will be an "empty" trait
/// implemented for all types.
#[cfg(feature = "serde1")]
pub trait SerializeAlias: Serialize {}

/// If the `serde1` feature is set, it acts as an alias for `Serialize` and is implemented for all
/// types which implement `Serialize`. If `serde1` is not set, it will be an "empty" trait
/// implemented for all types.
#[cfg(not(feature = "serde1"))]
pub trait SerializeAlias {}

#[cfg(feature = "serde1")]
impl<T> SerializeAlias for T where T: Serialize {}
#[cfg(not(feature = "serde1"))]
impl<T> SerializeAlias for T {}

/// If the `serde1` feature is set, it acts as an alias for `DeserializeOwned` and is implemented
/// for all types which implement `DeserializeOwned`. If `serde1` is not set, it will be an "empty"
/// trait implemented for all types.
#[cfg(feature = "serde1")]
pub trait DeserializeOwnedAlias: DeserializeOwned {}
/// If the `serde1` feature is set, it acts as an alias for `DeserializeOwned` and is implemented
/// for all types which implement `DeserializeOwned`. If `serde1` is not set, it will be an "empty"
/// trait implemented for all types.
#[cfg(not(feature = "serde1"))]
pub trait DeserializeOwnedAlias {}

#[cfg(feature = "serde1")]
impl<T> DeserializeOwnedAlias for T where T: DeserializeOwned {}
#[cfg(not(feature = "serde1"))]
impl<T> DeserializeOwnedAlias for T {}
