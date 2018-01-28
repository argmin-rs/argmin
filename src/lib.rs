// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Optimizaton toolbox
//!
//! TODOs
//!
//! * Stopping criterions which can be stacked, also return the reason why a computation terminated
//! * keep track of cost function values
//! * count the number of cost function / gradient evaluations and return them
//! * redesign how lower and upper bound are dealt with. making them optional should be better.
#![recursion_limit = "1024"]
#![cfg_attr(feature = "clippy", feature(plugin))]
#![cfg_attr(feature = "clippy", plugin(clippy))]
#![warn(missing_docs)]
#[macro_use]
extern crate error_chain;
extern crate ndarray;
extern crate ndarray_linalg;
extern crate num;
extern crate rand;

use std::default::Default;
use num::ToPrimitive;
use errors::*;
use parameter::ArgminParameter;
use result::ArgminResult;

/// Trait for cost function values
/// TODO: Do this with trait aliases once they work in rust.
pub trait ArgminCostValue: ToPrimitive + Copy + Default + PartialOrd {}
impl<T> ArgminCostValue for T
where
    T: ToPrimitive + Copy + Default + PartialOrd,
{
}

/// Trait every solve needs to implement (in the future)
pub trait ArgminSolver<'a> {
    /// Parameter vector
    // type A: ArgminParameter<Self::A>;
    type A: ArgminParameter;
    /// Cost value
    type B: ArgminCostValue;
    /// Hessian
    type C;
    /// Initial parameter(s)
    type D;
    /// Type of Problem (TODO: Trait!)
    type E;

    /// Initializes the solver and sets the state to its initial state
    // fn init(&mut self, &'a Problem<'a, Self::A, Self::B, Self::C>, &Self::D) -> Result<()>;
    fn init(&mut self, &'a Self::E, &Self::D) -> Result<()>;

    /// Moves forward by a single iteration
    fn next_iter(&mut self) -> Result<ArgminResult<Self::A, Self::B>>;

    /// Run initialization and iterations at once
    fn run(&mut self, &'a Self::E, &Self::D) -> Result<ArgminResult<Self::A, Self::B>>;

    /// Handles the stopping criteria
    fn terminate(&self) -> bool;
}

/// Definition of the return type of the solvers
pub mod result;

/// Traits for implementing parameter vectors
pub mod parameter;

/// Problem formulation
pub mod problem;

/// Operator
pub mod operator;

/// A set of test functions like Rosenbrock's function and so on.
pub mod testfunctions;

/// Backtracking line search
pub mod backtracking;

/// Simulated Annealing
pub mod sa;

/// Gradient Descent
pub mod gradientdescent;

/// Nelder Mead method
pub mod neldermead;

/// Newton method
pub mod newton;

/// Landweber algorithm
pub mod landweber;

/// Conjugate Gradient method
pub mod cg;

/// Errors using `error-chain`
mod errors;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
