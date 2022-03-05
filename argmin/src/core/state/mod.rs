// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! TODO: Documentation

pub mod iterstate;
pub mod linearprogramstate;

pub use iterstate::IterState;
pub use linearprogramstate::LinearProgramState;

use crate::core::{ArgminFloat, OpWrapper, TerminationReason};
use std::collections::HashMap;

/// Types implemeting this trait can be used to keep track of a solver's state
pub trait State {
    /// Type of Parameter vector
    type Param: Clone;
    /// Floating Point Precision
    type Float: ArgminFloat;

    /// Constructor
    fn new() -> Self;

    /// This method is called after each iteration and checks if the new parameter vector is better
    /// than the previous one. If so, it will update the current best parameter vector and current
    /// best cost function value.
    fn update(&mut self);

    /// Returns a reference to the best parameter vector
    fn get_best_param_ref(&self) -> Option<&Self::Param>;

    /// Returns maximum number of iterations
    fn get_max_iters(&self) -> u64;

    /// Increment the number of iterations by one
    fn increment_iter(&mut self);

    /// Returns current number of iterations
    fn get_iter(&self) -> u64;

    /// Returns current cost function value
    fn get_cost(&self) -> Self::Float;

    /// Returns best cost function value
    fn get_best_cost(&self) -> Self::Float;

    /// Returns target cost
    fn get_target_cost(&self) -> Self::Float;

    /// Set all function evaluation counts to the evaluation counts of another operator
    /// wrapped in `OpWrapper`.
    fn set_func_counts<O>(&mut self, op: &OpWrapper<O>);

    /// Return whether the algorithm has terminated or not
    fn terminated(&self) -> bool;

    /// Set termination reason
    #[must_use]
    fn termination_reason(self, termination_reason: TerminationReason) -> Self;

    /// Returns termination reason
    fn get_termination_reason(&self) -> TerminationReason;

    /// Set time required so far
    fn time(&mut self, time: Option<instant::Duration>) -> &mut Self;

    /// Get time required so far
    fn get_time(&self) -> Option<instant::Duration>;

    /// Returns iteration number where the last best parameter vector was found
    fn get_last_best_iter(&self) -> u64;

    /// Returns whether the current parameter vector is also the best parameter vector found so
    /// far.
    fn is_best(&self) -> bool;

    /// Returns currecnt cost function evaluation count
    fn get_func_counts(&self) -> &HashMap<String, u64>;
}
