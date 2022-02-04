// Copyright 2018-2022-2022 argmin developers
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
/// iteration state
mod iterstate;
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
/// Definition of termination reasons
mod termination;

pub use anyhow::Error;
pub use errors::ArgminError;
pub use executor::Executor;
pub use iterstate::{IterState, LinearProgramState, State};
pub use kv::ArgminKV;
pub use nooperator::{MinimalNoOperator, NoOperator};
use num_traits::{Float, FloatConst, FromPrimitive, ToPrimitive};
pub use observers::*;
pub use opwrapper::OpWrapper;
pub use result::ArgminResult;
#[cfg(feature = "serde1")]
use serde::{de::DeserializeOwned, Serialize};
#[cfg(feature = "serde1")]
pub use serialization::{load_checkpoint, ArgminCheckpoint, CheckpointMode};
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

/// This trait needs to be implemented for every operator/cost function.
///
/// It is required to implement the `apply` method, all others are optional and provide a default
/// implementation which is essentially returning an error which indicates that the method has not
/// been implemented. Those methods (`gradient` and `modify`) only need to be implemented if the
/// uses solver requires it.
pub trait ArgminOp {
    // TODO: Once associated type defaults are stable, it hopefully will be possible to define
    // default types for `Hessian` and `Jacobian`.
    /// Type of the parameter vector
    type Param: Clone + SerializeAlias + DeserializeOwnedAlias;
    /// Output of the operator
    type Output: Clone + SerializeAlias + DeserializeOwnedAlias;
    /// Type of Hessian
    type Hessian: Clone + SerializeAlias + DeserializeOwnedAlias;
    /// Type of Jacobian
    type Jacobian: Clone + SerializeAlias + DeserializeOwnedAlias;
    /// Precision of floats
    type Float: ArgminFloat;

    /// Applies the operator/cost function to parameters
    fn apply(&self, _param: &Self::Param) -> Result<Self::Output, Error> {
        Err(ArgminError::NotImplemented {
            text: "Method `apply` of ArgminOp trait not implemented!".to_string(),
        }
        .into())
    }

    /// Computes the gradient at the given parameters
    fn gradient(&self, _param: &Self::Param) -> Result<Self::Param, Error> {
        Err(ArgminError::NotImplemented {
            text: "Method `gradient` of ArgminOp trait not implemented!".to_string(),
        }
        .into())
    }

    /// Computes the Hessian at the given parameters
    fn hessian(&self, _param: &Self::Param) -> Result<Self::Hessian, Error> {
        Err(ArgminError::NotImplemented {
            text: "Method `hessian` of ArgminOp trait not implemented!".to_string(),
        }
        .into())
    }

    /// Computes the Hessian at the given parameters
    fn jacobian(&self, _param: &Self::Param) -> Result<Self::Jacobian, Error> {
        Err(ArgminError::NotImplemented {
            text: "Method `jacobian` of ArgminOp trait not implemented!".to_string(),
        }
        .into())
    }

    /// Modifies a parameter vector. Comes with a variable that indicates the "degree" of the
    /// modification.
    fn modify(&self, _param: &Self::Param, _extent: Self::Float) -> Result<Self::Param, Error> {
        Err(ArgminError::NotImplemented {
            text: "Method `modify` of ArgminOp trait not implemented!".to_string(),
        }
        .into())
    }
}

/// Problems which implement this trait can be used for linear programming solvers
pub trait LinearProgram {
    /// Type of the parameter vector
    type Param: Clone + SerializeAlias + DeserializeOwnedAlias;
    /// Precision of floats
    type Float: ArgminFloat;

    /// TODO c for linear programs
    /// Those three could maybe be merged into a single function; name unclear
    fn c(&self) -> Result<&[Self::Float], Error> {
        Err(ArgminError::NotImplemented {
            text: "Method `c` of ArgminOp trait not implemented!".to_string(),
        }
        .into())
    }

    /// TODO b for linear programs
    fn b(&self) -> Result<&[Self::Float], Error> {
        Err(ArgminError::NotImplemented {
            text: "Method `b` of ArgminOp trait not implemented!".to_string(),
        }
        .into())
    }

    /// TODO A for linear programs
    #[allow(non_snake_case)]
    fn A(&self) -> Result<&[Vec<Self::Float>], Error> {
        Err(ArgminError::NotImplemented {
            text: "Method `A` of ArgminOp trait not implemented!".to_string(),
        }
        .into())
    }
}

/// Solver
///
/// Every solver needs to implement this trait.
pub trait Solver<I: State>: SerializeAlias {
    /// Name of the solver
    const NAME: &'static str = "UNDEFINED";

    /// Computes one iteration of the algorithm.
    fn next_iter(
        &mut self,
        op: &mut OpWrapper<I::Operator>,
        state: &mut I,
    ) -> Result<ArgminIterData<I>, Error>;

    /// Initializes the algorithm
    ///
    /// This is executed before any iterations are performed. It can be used to perform
    /// precomputations. The default implementation corresponds to doing nothing.
    fn init(
        &mut self,
        _op: &mut OpWrapper<I::Operator>,
        _state: &mut I,
    ) -> Result<Option<ArgminIterData<I>>, Error> {
        Ok(None)
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

/// The datastructure which is returned by the `next_iter` method of the `Solver` trait.
///
/// TODO: Rename to IterResult?
#[derive(Clone, Debug, Default)]
pub struct ArgminIterData<I: State> {
    /// Current parameter vector
    param: Option<I::Param>,
    /// Current cost function value
    cost: Option<I::Float>,
    /// Current gradient
    grad: Option<I::Param>,
    /// Current Hessian
    hessian: Option<I::Hessian>,
    /// Current inverse Hessian
    inv_hessian: Option<I::Hessian>,
    /// Current Jacobian
    jacobian: Option<I::Jacobian>,
    /// Current population
    population: Option<Vec<(I::Param, I::Float)>>,
    /// terminationreason
    termination_reason: Option<TerminationReason>,
    /// Key value pairs which are used to provide additional information for the Observers
    kv: ArgminKV,
}

// TODO: Many clones are necessary in the getters.. maybe a complete "deconstruct" method would be
// better?
impl<I: State> ArgminIterData<I> {
    /// Constructor
    pub fn new() -> Self {
        ArgminIterData {
            param: None,
            cost: None,
            grad: None,
            hessian: None,
            inv_hessian: None,
            jacobian: None,
            termination_reason: None,
            population: None,
            kv: make_kv!(),
        }
    }

    /// Set parameter vector
    #[must_use]
    pub fn param(mut self, param: I::Param) -> Self {
        self.param = Some(param);
        self
    }

    /// Set cost function value
    #[must_use]
    pub fn cost(mut self, cost: I::Float) -> Self {
        self.cost = Some(cost);
        self
    }

    /// Set gradient
    #[must_use]
    pub fn grad(mut self, grad: I::Param) -> Self {
        self.grad = Some(grad);
        self
    }

    /// Set Hessian
    #[must_use]
    pub fn hessian(mut self, hessian: I::Hessian) -> Self {
        self.hessian = Some(hessian);
        self
    }

    /// Set inverse Hessian
    #[must_use]
    pub fn inv_hessian(mut self, inv_hessian: I::Hessian) -> Self {
        self.inv_hessian = Some(inv_hessian);
        self
    }

    /// Set Jacobian
    #[must_use]
    pub fn jacobian(mut self, jacobian: I::Jacobian) -> Self {
        self.jacobian = Some(jacobian);
        self
    }

    /// Set Population
    #[must_use]
    pub fn population(mut self, population: Vec<(I::Param, I::Float)>) -> Self {
        self.population = Some(population);
        self
    }

    /// Adds an `ArgminKV`
    #[must_use]
    pub fn kv(mut self, kv: ArgminKV) -> Self {
        self.kv = kv;
        self
    }

    /// Set termination reason
    #[must_use]
    pub fn termination_reason(mut self, reason: TerminationReason) -> Self {
        self.termination_reason = Some(reason);
        self
    }

    /// Get parameter vector
    pub fn get_param(&self) -> Option<I::Param> {
        self.param.clone()
    }

    /// Get cost function value
    pub fn get_cost(&self) -> Option<I::Float> {
        self.cost
    }

    /// Get gradient
    pub fn get_grad(&self) -> Option<I::Param> {
        self.grad.clone()
    }

    /// Get Hessian
    pub fn get_hessian(&self) -> Option<I::Hessian> {
        self.hessian.clone()
    }

    /// Get inverse Hessian
    pub fn get_inv_hessian(&self) -> Option<I::Hessian> {
        self.inv_hessian.clone()
    }

    /// Get Jacobian
    pub fn get_jacobian(&self) -> Option<I::Jacobian> {
        self.jacobian.clone()
    }

    /// Get reference to population
    pub fn get_population(&self) -> Option<&Vec<(I::Param, I::Float)>> {
        match &self.population {
            Some(population) => Some(population),
            None => None,
        }
    }

    /// Get termination reason
    pub fn get_termination_reason(&self) -> Option<TerminationReason> {
        self.termination_reason
    }

    /// Return KV
    pub fn get_kv(&self) -> ArgminKV {
        self.kv.clone()
    }
}

/// Defines a common interface for line search methods.
pub trait ArgminLineSearch<P, F> {
    /// Set the search direction
    fn set_search_direction(&mut self, direction: P);

    /// Set the initial step length
    fn set_init_alpha(&mut self, step_length: F) -> Result<(), Error>;
}

/// Defines a common interface to methods which calculate approximate steps for trust region
/// methods.
pub trait ArgminTrustRegion<F> {
    /// Set the initial step length
    fn set_radius(&mut self, radius: F);
}
//
/// Common interface for beta update methods (Nonlinear-CG)
pub trait ArgminNLCGBetaUpdate<T, F: ArgminFloat>: SerializeAlias {
    /// Update beta
    /// Parameter 1: \nabla f_k
    /// Parameter 2: \nabla f_{k+1}
    /// Parameter 3: p_k
    fn update(&self, nabla_f_k: &T, nabla_f_k_p_1: &T, p_k: &T) -> F;
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
