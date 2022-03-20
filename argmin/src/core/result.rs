// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::{ArgminFloat, Problem, State};
use num_traits::{Float, FromPrimitive};
use std::cmp::Ordering;
use std::fmt;

/// Result of an optimization returned by after running an `Executor`.
///
/// Consists of the problem and the final state of the solver.
/// Both can be accessed via deconstructing or via the methods
/// [`problem`](`OptimizationResult::problem`) and [`state`](`OptimizationResult::state`).
#[derive(Clone)]
pub struct OptimizationResult<O, I> {
    /// Problem
    pub problem: Problem<O>,
    /// Iteration state
    pub state: I,
}

impl<O, I> OptimizationResult<O, I> {
    /// Constructs a new instance of `OptimizationResult` from a `problem` and a `state`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{Problem, OptimizationResult, IterState, State};
    /// # use argmin::core::test_utils::TestProblem;
    /// #
    /// # type Rosenbrock = TestProblem;
    /// #
    /// let rosenbrock = Rosenbrock::new();
    /// let state: IterState<Vec<f64>, (), (), (), f64> = IterState::new();
    ///
    /// let result = OptimizationResult::new(Problem::new(rosenbrock), state);
    /// # let OptimizationResult { mut problem, state } = result;
    /// # assert_eq!(problem.take_problem().unwrap(), TestProblem::new());
    /// ```
    pub fn new(problem: Problem<O>, state: I) -> Self {
        OptimizationResult { problem, state }
    }

    /// Returns a reference to the stored problem.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{Problem, OptimizationResult, IterState, State};
    /// #
    /// # struct Rosenbrock {}
    /// #
    /// # let state: IterState<Vec<f64>, (), (), (), f64> = IterState::new();
    /// #
    /// # let result = OptimizationResult::new(Problem::new(Rosenbrock {}), state);
    /// #
    /// let problem: &Problem<Rosenbrock> = result.problem();
    /// ```
    pub fn problem(&self) -> &Problem<O> {
        &self.problem
    }

    /// Returns a reference to the stored state.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{Problem, OptimizationResult, IterState, State};
    /// #
    /// # struct Rosenbrock {}
    /// #
    /// # let state: IterState<Vec<f64>, (), (), (), f64> = IterState::new();
    /// #
    /// # let result = OptimizationResult::new(Problem::new(Rosenbrock {}), state);
    /// #
    /// let state: &IterState<Vec<f64>, (), (), (), f64> = result.state();
    /// ```
    pub fn state(&self) -> &I {
        &self.state
    }
}

impl<O, I> std::fmt::Display for OptimizationResult<O, I>
where
    I: State,
    I::Param: fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(f, "OptimizationResult:")?;
        writeln!(
            f,
            "    param (best):  {}",
            if let Some(best_param) = self.state.get_best_param_ref() {
                format!("{:?}", best_param)
            } else {
                String::from("None")
            }
        )?;
        writeln!(f, "    cost (best):   {}", self.state.get_best_cost())?;
        writeln!(f, "    iters (best):  {}", self.state.get_last_best_iter())?;
        writeln!(f, "    iters (total): {}", self.state.get_iter())?;
        writeln!(
            f,
            "    termination:   {}",
            self.state.get_termination_reason()
        )?;
        if let Some(time) = self.state.get_time() {
            writeln!(f, "    time:          {:?}", time)?;
        }
        Ok(())
    }
}

impl<O, I: State> PartialEq for OptimizationResult<O, I>
where
    I::Float: ArgminFloat,
{
    /// Two `OptimizationResult`s are equal if the absolute of the difference between their best
    /// cost values is smaller than epsilon.
    fn eq(&self, other: &OptimizationResult<O, I>) -> bool {
        (self.state.get_best_cost() - other.state.get_best_cost()).abs() < I::Float::epsilon()
    }
}

impl<O, I: State> Eq for OptimizationResult<O, I> {}

impl<O, I: State> Ord for OptimizationResult<O, I> {
    /// Two `OptimizationResult`s are equal if the absolute of the difference between their best
    /// cost values is smaller than epsilon.
    /// Else, an `OptimizationResult` is better if the best cost function value is strictly better
    /// than the other's.
    fn cmp(&self, other: &OptimizationResult<O, I>) -> Ordering {
        let t = self.state.get_best_cost() - other.state.get_best_cost();
        if t.abs() < I::Float::epsilon() {
            Ordering::Equal
        } else if t > I::Float::from_f64(0.0).unwrap() {
            Ordering::Greater
        } else {
            Ordering::Less
        }
    }
}

impl<O, I: State> PartialOrd for OptimizationResult<O, I> {
    /// Two `OptimizationResult`s are equal if the absolute of the difference between their best
    /// cost values is smaller than epsilon.
    /// Else, an `OptimizationResult` is better if the best cost function value is strictly better
    /// than the other's.
    fn partial_cmp(&self, other: &OptimizationResult<O, I>) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{test_utils::TestProblem, IterState};

    send_sync_test!(
        optimizationresult,
        OptimizationResult<TestProblem, IterState<(), (), (), (), f64>>
    );

    // TODO: More tests, in particular the checking that the output is as intended.
}
