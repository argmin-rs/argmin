//! Optimizaton toolbox
//!
//! TODO
#![recursion_limit = "1024"]
#![cfg_attr(feature = "clippy", feature(plugin))]
#![cfg_attr(feature = "clippy", plugin(clippy))]
// #![warn(missing_docs)]
#[macro_use]
extern crate error_chain;
extern crate num;
extern crate rand;

/// Trait for forward operators
pub trait ArgminOperator {}

/// Trait for cost functions
pub trait ArgminCost {}

/// Definition of the return type of the solvers
pub mod result;

/// Traits for implementing parameter vectors
pub mod parameter;

/// Problem formulation
pub mod problem;

/// A set of test functions like Rosenbrock's function and so on.
pub mod testfunctions;

/// Simulated Annealing
pub mod sa;

/// Gradient Descent
// pub mod gradientdescent;

/// Errors using `error-chain`
mod errors;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
