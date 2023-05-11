// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::{
    ArgminFloat, CostFunction, DeserializeOwnedAlias, Error, Executor, Gradient, IterState,
    LineSearch, OptimizationResult, Problem, SerializeAlias, Solver, KV,
};
use argmin_math::ArgminMul;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// # Steepest descent
///
/// Iteratively takes steps in the direction of the strongest negative gradient. In each iteration,
/// a line search is used to obtain an appropriate step length.
///
/// ## Requirements on the optimization problem
///
/// The optimization problem is required to implement [`CostFunction`] and [`Gradient`].
///
/// ## Reference
///
/// Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct SteepestDescent<L> {
    /// line search
    linesearch: L,
}

impl<L> SteepestDescent<L> {
    /// Construct a new instance of [`SteepestDescent`]
    ///
    /// Requires a line search.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::gradientdescent::SteepestDescent;
    /// # let linesearch = ();
    /// let sd = SteepestDescent::new(linesearch);
    /// ```
    pub fn new(linesearch: L) -> Self {
        SteepestDescent { linesearch }
    }
}

impl<O, L, P, G, F> Solver<O, IterState<P, G, (), (), F>> for SteepestDescent<L>
where
    O: CostFunction<Param = P, Output = F> + Gradient<Param = P, Gradient = G>,
    P: Clone + SerializeAlias + DeserializeOwnedAlias,
    G: Clone + SerializeAlias + DeserializeOwnedAlias + ArgminMul<F, G>,
    L: Clone + LineSearch<G, F> + Solver<O, IterState<P, G, (), (), F>>,
    F: ArgminFloat,
{
    const NAME: &'static str = "Steepest Descent";

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        mut state: IterState<P, G, (), (), F>,
    ) -> Result<(IterState<P, G, (), (), F>, Option<KV>), Error> {
        let param_new = state.take_param().ok_or_else(argmin_error_closure!(
            NotInitialized,
            concat!(
                "`SteepestDescent` requires an initial parameter vector. ",
                "Please provide an initial guess via `Executor`s `configure` method."
            )
        ))?;
        let new_cost = problem.cost(&param_new)?;
        let new_grad = problem.gradient(&param_new)?;

        self.linesearch
            .search_direction(new_grad.mul(&(float!(-1.0))));

        // Run line search
        let OptimizationResult {
            problem: line_problem,
            state: mut linesearch_state,
            ..
        } = Executor::new(
            problem.take_problem().ok_or_else(argmin_error_closure!(
                PotentialBug,
                "`SteepestDescent`: Failed to take `problem` for line search"
            ))?,
            self.linesearch.clone(),
        )
        .configure(|config| config.param(param_new).gradient(new_grad).cost(new_cost))
        .ctrlc(false)
        .run()?;

        // Get back problem and function evaluation counts
        problem.consume_problem(line_problem);

        Ok((
            state
                .param(
                    linesearch_state
                        .take_param()
                        .ok_or_else(argmin_error_closure!(
                            PotentialBug,
                            "`GradientDescent`: No `param` returned by line search"
                        ))?,
                )
                .cost(linesearch_state.get_cost()),
            None,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::test_utils::TestProblem;
    use crate::core::{ArgminError, State};
    use crate::solver::linesearch::{
        condition::ArmijoCondition, BacktrackingLineSearch, MoreThuenteLineSearch,
    };
    use crate::test_trait_impl;
    use approx::assert_relative_eq;

    test_trait_impl!(
        steepest_descent,
        SteepestDescent<MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64>>
    );

    #[test]
    fn test_new() {
        let linesearch: BacktrackingLineSearch<Vec<f64>, Vec<f64>, ArmijoCondition<f64>, f64> =
            BacktrackingLineSearch::new(ArmijoCondition::new(0.2).unwrap());
        let SteepestDescent { linesearch: ls } = SteepestDescent::new(linesearch.clone());
        assert_eq!(ls, linesearch);
    }

    #[test]
    fn test_next_iter_param_not_initialized() {
        let linesearch: BacktrackingLineSearch<Vec<f64>, Vec<f64>, ArmijoCondition<f64>, f64> =
            BacktrackingLineSearch::new(ArmijoCondition::new(0.2).unwrap());
        let mut sd = SteepestDescent::new(linesearch);
        let res = sd.next_iter(&mut Problem::new(TestProblem::new()), IterState::new());
        assert_error!(
            res,
            ArgminError,
            concat!(
                "Not initialized: \"`SteepestDescent` requires an initial parameter vector. ",
                "Please provide an initial guess via `Executor`s `configure` method.\""
            )
        );
    }

    #[test]
    fn test_next_iter_regression() {
        struct SDProblem {}

        impl CostFunction for SDProblem {
            type Param = Vec<f64>;
            type Output = f64;

            fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
                Ok(p[0].powi(2) + p[1].powi(2))
            }
        }

        impl Gradient for SDProblem {
            type Param = Vec<f64>;
            type Gradient = Vec<f64>;

            fn gradient(&self, p: &Self::Param) -> Result<Self::Param, Error> {
                Ok(vec![2.0 * p[0], 2.0 * p[1]])
            }
        }

        let linesearch: BacktrackingLineSearch<Vec<f64>, Vec<f64>, ArmijoCondition<f64>, f64> =
            BacktrackingLineSearch::new(ArmijoCondition::new(0.2).unwrap());
        let mut sd = SteepestDescent::new(linesearch);
        let (state, kv) = sd
            .next_iter(
                &mut Problem::new(SDProblem {}),
                IterState::new().param(vec![1.0, 2.0]),
            )
            .unwrap();

        assert!(kv.is_none());

        assert_relative_eq!(
            state.param.as_ref().unwrap()[0],
            -0.4580000000000002,
            epsilon = f64::EPSILON
        );
        assert_relative_eq!(
            state.param.as_ref().unwrap()[1],
            -0.9160000000000004,
            epsilon = f64::EPSILON
        );
        assert_relative_eq!(state.cost, 1.048820000000001, epsilon = f64::EPSILON);
    }
}
