// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Steepest Descent method
//!
//! [SteepestDescent](struct.SteepestDescent.html)
//!
//! # References:
//!
//! \[0\] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
//! Springer. ISBN 0-387-30303-0.

use crate::core::{
    ArgminFloat, CostFunction, DeserializeOwnedAlias, Error, Executor, Gradient, IterState,
    LineSearch, OptimizationResult, Problem, SerializeAlias, Solver, KV,
};
use argmin_math::ArgminMul;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// Steepest descent iteratively takes steps in the direction of the strongest negative gradient.
/// In each iteration, a line search is employed to obtain an appropriate step length.
///
/// # References:
///
/// \[0\] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct SteepestDescent<L> {
    /// line search
    linesearch: L,
}

impl<L> SteepestDescent<L> {
    /// Constructor
    pub fn new(linesearch: L) -> Self {
        SteepestDescent { linesearch }
    }
}

impl<O, L, P, G, F> Solver<O, IterState<P, G, (), (), F>> for SteepestDescent<L>
where
    O: CostFunction<Param = P, Output = F> + Gradient<Param = P, Gradient = G>,
    P: Clone + SerializeAlias + DeserializeOwnedAlias,
    G: Clone + SerializeAlias + DeserializeOwnedAlias + ArgminMul<F, P>,
    L: Clone + LineSearch<P, F> + Solver<O, IterState<P, G, (), (), F>>,
    F: ArgminFloat,
{
    const NAME: &'static str = "Steepest Descent";

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        mut state: IterState<P, G, (), (), F>,
    ) -> Result<(IterState<P, G, (), (), F>, Option<KV>), Error> {
        let param_new = state.take_param().unwrap();
        let new_cost = problem.cost(&param_new)?;
        let new_grad = problem.gradient(&param_new)?;

        self.linesearch
            .set_search_direction(new_grad.mul(&(F::from_f64(-1.0).unwrap())));

        // Run solver
        let OptimizationResult {
            problem: line_problem,
            state: mut linesearch_state,
            ..
        } = Executor::new(problem.take_problem().unwrap(), self.linesearch.clone())
            .configure(|config| config.param(param_new).grad(new_grad).cost(new_cost))
            .ctrlc(false)
            .run()?;

        // Get back problem and function evaluation counts
        problem.consume_problem(line_problem);

        Ok((
            state
                .param(linesearch_state.take_param().unwrap())
                .cost(linesearch_state.get_cost()),
            None,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::linesearch::MoreThuenteLineSearch;
    use crate::test_trait_impl;

    test_trait_impl!(
        steepest_descent,
        SteepestDescent<MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64>>
    );
}
