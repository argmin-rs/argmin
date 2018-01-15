//! Optimizaton toolbox
//!
//! TODOs
//!
//! * Stopping criterions which can be stacked, also return the reason why a computation terminated
//! * keep track of cost function values
//! * count the number of cost function / gradient evaluations and return them
#![recursion_limit = "1024"]
#![cfg_attr(feature = "clippy", feature(plugin))]
#![cfg_attr(feature = "clippy", plugin(clippy))]
#![warn(missing_docs)]
#[macro_use]
extern crate error_chain;
extern crate num;
extern crate rand;

/// Trait for cost function values
/// TODO: Do this with trait aliases once they work in rust.
pub trait ArgminCostValue: num::Float + num::FromPrimitive + PartialOrd {}
impl<T> ArgminCostValue for T
where
    T: num::Float + num::FromPrimitive + num::Num + PartialOrd,
{
}

/// Definition of the return type of the solvers
pub mod result;

/// Traits for implementing parameter vectors
pub mod parameter;

/// Problem formulation
pub mod problem;

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

/// Errors using `error-chain`
mod errors;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
