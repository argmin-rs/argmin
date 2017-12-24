//! Optimizaton toolbox
//!
//! TODO
#![recursion_limit = "1024"]
#![cfg_attr(feature = "clippy", feature(plugin))]
#![cfg_attr(feature = "clippy", plugin(clippy))]
#![warn(missing_docs)]
#[macro_use]
extern crate error_chain;

mod errors {
    error_chain!{}
}

use errors::*;

/// Trait for forward operators
pub trait ArgminOperator {}

/// Trait for cost functions
pub trait ArgminCost {}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
