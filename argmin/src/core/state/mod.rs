// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

pub mod iterstate;
pub mod linearprogramstate;
pub mod populationstate;

pub use iterstate::IterState;
pub use linearprogramstate::LinearProgramState;
pub use populationstate::PopulationState;

use crate::core::{ArgminFloat, Problem, TerminationReason, TerminationStatus};
use std::collections::HashMap;

/// Minimal interface which struct used for managing state in solvers have to implement.
///
/// These methods expose basic information about the state which is needed in
/// [`Executor`](`crate::core::Executor`) and
/// [`OptimizationResult`](`crate::core::OptimizationResult`) but can also be useful in
/// [`observers`](`crate::core::observers`).
///
/// The struct implementing this trait should keep track of
/// * the current parameter vector
/// * the cost associated with the current parameter vector
/// * the current best parameter vector
/// * the cost associated with the current best parameter vector
/// * the iteration number where the last best parameter vector was found
/// * the target cost function value (If this value is reached, the optimization will be stopped).
///   Set this to `Self::Float::NEG_INFINITY` if not relevant.
/// * the current number of iterations
/// * how often each function of the problem has been called
/// * the time required since the beginning of the optimization until the current point in time
/// * the status of optimization execution ([`TerminationStatus`])
///
/// Since the state in general changes for each iteration, "current" refers to the current
/// iteration.
///
/// [`State::Param`] indicates the type of the parameter vector while [`State::Float`] indicates
/// the precision of floating point operations. Any type implementing [`ArgminFloat`] can be used
/// for this (so far f32 and f64).
pub trait State {
    /// Type of parameter vector
    type Param;
    /// Floating point precision (f32 or f64)
    type Float: ArgminFloat;

    /// Construct a new state
    fn new() -> Self;

    /// This method is called after each iteration and checks if the new parameter vector is better
    /// than the previous one. If so, it will update the current best parameter vector and current
    /// best cost function value.
    ///
    /// For methods where the cost function value is unknown, it is advised to assume that every
    /// new parameter vector is better than the previous one.
    fn update(&mut self);

    /// Returns a reference to the current parameter vector
    fn get_param(&self) -> Option<&Self::Param>;

    /// Returns a reference to the current best parameter vector
    fn get_best_param(&self) -> Option<&Self::Param>;

    /// Returns maximum number of iterations that are to be performed
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
    /// wrapped in `Problem`.
    fn func_counts<O>(&mut self, problem: &Problem<O>);

    /// Returns current cost function evaluation count
    fn get_func_counts(&self) -> &HashMap<String, u64>;

    /// Set time required since the beginning of the optimization until the current iteration
    fn time(&mut self, time: Option<instant::Duration>) -> &mut Self;

    /// Get time passed since the beginning of the optimization until the current iteration
    fn get_time(&self) -> Option<instant::Duration>;

    /// Returns iteration number where the last best parameter vector was found
    fn get_last_best_iter(&self) -> u64;

    /// Returns whether the current parameter vector is also the best parameter vector found so
    /// far.
    fn is_best(&self) -> bool;

    /// Sets the termination status to [`Terminated`](`TerminationStatus::Terminated`) with the given reason
    #[must_use]
    fn terminate_with(self, termination_reason: TerminationReason) -> Self;

    /// Returns termination status.
    fn get_termination_status(&self) -> TerminationStatus;

    /// Returns the termination reason if terminated, otherwise None.
    fn get_termination_reason(&self) -> Option<TerminationReason>;

    /// Return whether the algorithm has terminated or not
    fn terminated(&self) -> bool {
        matches!(
            self.get_termination_status(),
            TerminationStatus::Terminated(_)
        )
    }
}
