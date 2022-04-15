// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! * [Backtracking line search](struct.BacktrackingLineSearch.html)

use crate::core::{
    ArgminFloat, CostFunction, Error, Gradient, IterState, LineSearch, Problem, SerializeAlias,
    Solver, State, TerminationReason, KV,
};
use crate::solver::linesearch::condition::*;
use argmin_math::ArgminScaledAdd;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// The Backtracking line search is a simple method to find a step length which obeys the Armijo
/// (sufficient decrease) condition.
///
/// # References:
///
/// \[0\] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
///
/// \[1\] Wikipedia: <https://en.wikipedia.org/wiki/Backtracking_line_search>
#[derive(Clone, Eq, PartialEq, Debug)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct BacktrackingLineSearch<P, G, L, F> {
    /// initial parameter vector
    init_param: Option<P>,
    /// initial cost
    init_cost: F,
    /// initial gradient
    init_grad: Option<G>,
    /// Search direction
    search_direction: Option<P>,
    /// Contraction factor rho
    rho: F,
    /// Stopping condition
    condition: L,
    /// alpha
    alpha: F,
}

impl<P, G, L, F> BacktrackingLineSearch<P, G, L, F>
where
    F: ArgminFloat,
{
    /// Constructor
    pub fn new(condition: L) -> Self {
        BacktrackingLineSearch {
            init_param: None,
            init_cost: F::infinity(),
            init_grad: None,
            search_direction: None,
            rho: F::from_f64(0.9).unwrap(),
            condition,
            alpha: F::from_f64(1.0).unwrap(),
        }
    }

    /// Set rho
    pub fn rho(mut self, rho: F) -> Result<Self, Error> {
        if rho <= F::from_f64(0.0).unwrap() || rho >= F::from_f64(1.0).unwrap() {
            return Err(argmin_error!(
                InvalidParameter,
                "BacktrackingLineSearch: Contraction factor rho must be in (0, 1)."
            ));
        }
        self.rho = rho;
        Ok(self)
    }
}

impl<P, G, L, F> LineSearch<P, F> for BacktrackingLineSearch<P, G, L, F>
where
    F: ArgminFloat,
{
    /// Set search direction
    fn search_direction(&mut self, search_direction: P) {
        self.search_direction = Some(search_direction);
    }

    /// Set initial alpha value
    fn initial_step_length(&mut self, alpha: F) -> Result<(), Error> {
        if alpha <= F::from_f64(0.0).unwrap() {
            return Err(argmin_error!(
                InvalidParameter,
                "LineSearch: Inital alpha must be > 0."
            ));
        }
        self.alpha = alpha;
        Ok(())
    }
}

impl<P, G, L, F> BacktrackingLineSearch<P, G, L, F>
where
    P: ArgminScaledAdd<P, F, P>,
    L: LineSearchCondition<P, G, F>,
    IterState<P, G, (), (), F>: State<Float = F>,
    F: ArgminFloat,
{
    fn backtracking_step<O>(
        &self,
        problem: &mut Problem<O>,
        state: IterState<P, G, (), (), F>,
    ) -> Result<IterState<P, G, (), (), F>, Error>
    where
        O: CostFunction<Param = P, Output = F> + Gradient<Param = P, Gradient = G>,
    {
        let new_param = self
            .init_param
            .as_ref()
            .unwrap()
            .scaled_add(&self.alpha, self.search_direction.as_ref().unwrap());

        let cur_cost = problem.cost(&new_param)?;

        let out = if self.condition.requires_current_gradient() {
            state
                .grad(problem.gradient(&new_param)?)
                .param(new_param)
                .cost(cur_cost)
        } else {
            state.param(new_param).cost(cur_cost)
        };

        Ok(out)
    }
}

impl<O, P, G, L, F> Solver<O, IterState<P, G, (), (), F>> for BacktrackingLineSearch<P, G, L, F>
where
    P: Clone + SerializeAlias + ArgminScaledAdd<P, F, P>,
    G: SerializeAlias + ArgminScaledAdd<P, F, P>,
    O: CostFunction<Param = P, Output = F> + Gradient<Param = P, Gradient = G>,
    L: LineSearchCondition<P, G, F> + SerializeAlias,
    F: ArgminFloat,
{
    const NAME: &'static str = "Backtracking line search";

    fn init(
        &mut self,
        problem: &mut Problem<O>,
        mut state: IterState<P, G, (), (), F>,
    ) -> Result<(IterState<P, G, (), (), F>, Option<KV>), Error> {
        let init_param = state.param.clone().unwrap();
        let cost = state.cost;
        self.init_cost = if cost == F::infinity() {
            problem.cost(&init_param)?
        } else {
            cost
        };

        let init_grad = state
            .take_grad()
            .map(Result::Ok)
            .unwrap_or_else(|| problem.gradient(&init_param))?;

        if self.search_direction.is_none() {
            return Err(argmin_error!(
                NotInitialized,
                "BacktrackingLineSearch: search_direction must be set."
            ));
        }

        self.init_param = Some(init_param);
        self.init_grad = Some(init_grad);
        let state = self.backtracking_step(problem, state)?;
        Ok((state, None))
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        state: IterState<P, G, (), (), F>,
    ) -> Result<(IterState<P, G, (), (), F>, Option<KV>), Error> {
        self.alpha = self.alpha * self.rho;
        let state = self.backtracking_step(problem, state)?;
        Ok((state, None))
    }

    fn terminate(&mut self, state: &IterState<P, G, (), (), F>) -> TerminationReason {
        if self.condition.evaluate_condition(
            state.cost,
            state.get_grad(),
            self.init_cost,
            self.init_grad.as_ref().unwrap(),
            self.search_direction.as_ref().unwrap(),
            self.alpha,
        ) {
            TerminationReason::LineSearchConditionMet
        } else {
            TerminationReason::NotTerminated
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_error;
    use crate::core::{test_utils::TestProblem, ArgminError, Executor, State};
    use crate::test_trait_impl;
    use approx::assert_relative_eq;
    use num_traits::Float;

    #[derive(Debug, Clone)]
    struct BTTestProblem {}

    impl CostFunction for BTTestProblem {
        type Param = Vec<f64>;
        type Output = f64;

        fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
            Ok(p[0].powi(2) + p[1].powi(2))
        }
    }

    impl Gradient for BTTestProblem {
        type Param = Vec<f64>;
        type Gradient = Vec<f64>;

        fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
            Ok(vec![2.0 * p[0], 2.0 * p[1]])
        }
    }

    test_trait_impl!(backtrackinglinesearch,
                    BacktrackingLineSearch<TestProblem, Vec<f64>, ArmijoCondition<f64>, f64>);

    #[test]
    fn test_new() {
        let c: f64 = 0.01;
        let armijo = ArmijoCondition::new(c).unwrap();
        let ls: BacktrackingLineSearch<Vec<f64>, Vec<f64>, ArmijoCondition<f64>, f64> =
            BacktrackingLineSearch::new(armijo);

        assert_eq!(ls.init_param, None);
        assert!(ls.init_cost.is_infinite());
        assert!(ls.init_cost.is_sign_positive());
        assert_eq!(ls.init_grad, None);
        assert_eq!(ls.search_direction, None);
        assert_eq!(ls.rho.to_ne_bytes(), 0.9f64.to_ne_bytes());
        assert_eq!(ls.alpha.to_ne_bytes(), 1.0f64.to_ne_bytes());
    }

    #[test]
    fn test_rho() {
        let c: f64 = 0.01;
        let armijo = ArmijoCondition::new(c).unwrap();
        let ls: BacktrackingLineSearch<Vec<f64>, Vec<f64>, ArmijoCondition<f64>, f64> =
            BacktrackingLineSearch::new(armijo);

        assert_error!(
            ls.rho(1.0f64),
            ArgminError,
            "Invalid parameter: \"BacktrackingLineSearch: Contraction factor rho must be in (0, 1).\""
        );

        let c: f64 = 0.01;
        let armijo = ArmijoCondition::new(c).unwrap();
        let ls: BacktrackingLineSearch<Vec<f64>, Vec<f64>, ArmijoCondition<f64>, f64> =
            BacktrackingLineSearch::new(armijo);

        assert_error!(
            ls.rho(0.0f64),
            ArgminError,
            "Invalid parameter: \"BacktrackingLineSearch: Contraction factor rho must be in (0, 1).\""
        );

        let c: f64 = 0.01;
        let armijo = ArmijoCondition::new(c).unwrap();
        let ls: BacktrackingLineSearch<Vec<f64>, Vec<f64>, ArmijoCondition<f64>, f64> =
            BacktrackingLineSearch::new(armijo);

        assert!(ls.rho(0.0f64 + f64::EPSILON).is_ok());

        let c: f64 = 0.01;
        let armijo = ArmijoCondition::new(c).unwrap();
        let ls: BacktrackingLineSearch<Vec<f64>, Vec<f64>, ArmijoCondition<f64>, f64> =
            BacktrackingLineSearch::new(armijo);

        assert!(ls.rho(1.0f64 - f64::EPSILON).is_ok());
    }

    #[test]
    fn test_search_direction() {
        let c: f64 = 0.01;
        let armijo = ArmijoCondition::new(c).unwrap();
        let mut ls: BacktrackingLineSearch<Vec<f64>, Vec<f64>, ArmijoCondition<f64>, f64> =
            BacktrackingLineSearch::new(armijo);
        ls.search_direction(vec![1.0f64, 1.0]);

        assert_eq!(ls.search_direction, Some(vec![1.0f64, 1.0]));
    }

    #[test]
    fn test_initial_step_length() {
        let c: f64 = 0.01;
        let armijo = ArmijoCondition::new(c).unwrap();
        let mut ls: BacktrackingLineSearch<Vec<f64>, Vec<f64>, ArmijoCondition<f64>, f64> =
            BacktrackingLineSearch::new(armijo);

        assert!(ls.initial_step_length(f64::EPSILON).is_ok());

        assert_error!(
            ls.initial_step_length(0.0f64),
            ArgminError,
            "Invalid parameter: \"LineSearch: Inital alpha must be > 0.\""
        );
    }

    #[test]
    fn test_step_armijo() {
        use crate::core::Problem;

        let prob = BTTestProblem {};

        let c: f64 = 0.01;
        let armijo = ArmijoCondition::new(c).unwrap();
        let mut ls: BacktrackingLineSearch<Vec<f64>, Vec<f64>, ArmijoCondition<f64>, f64> =
            BacktrackingLineSearch::new(armijo);

        ls.init_param = Some(vec![-1.0, 0.0]);
        ls.init_cost = f64::infinity();
        ls.init_grad = Some(vec![-2.0, 0.0]);
        ls.search_direction(vec![2.0f64, 0.0]);
        ls.initial_step_length(0.8).unwrap();

        let data = ls.backtracking_step(&mut Problem::new(prob), IterState::new());
        assert!(data.is_ok());

        let param = data.as_ref().unwrap().get_param().unwrap();
        let cost = data.as_ref().unwrap().get_cost();
        assert_relative_eq!(param[0], 0.6, epsilon = f64::EPSILON);
        assert_relative_eq!(param[1], 0.0, epsilon = f64::EPSILON);
        assert_relative_eq!(cost, 0.6f64.powi(2), epsilon = f64::EPSILON);

        assert!(data.as_ref().unwrap().get_grad().is_none());
    }

    #[test]
    fn test_step_wolfe() {
        // Wolfe, in contrast to Armijo, requires the current gradient. This test makes sure that
        // the implementation of the backtracking linesearch properly considers this.
        use crate::core::Problem;

        let prob = BTTestProblem {};

        let c1: f64 = 0.01;
        let c2: f64 = 0.9;
        let wolfe = WolfeCondition::new(c1, c2).unwrap();
        let mut ls: BacktrackingLineSearch<Vec<f64>, Vec<f64>, WolfeCondition<f64>, f64> =
            BacktrackingLineSearch::new(wolfe);

        ls.init_param = Some(vec![-1.0, 0.0]);
        ls.init_cost = f64::infinity();
        ls.init_grad = Some(vec![-2.0, 0.0]);
        ls.search_direction(vec![2.0f64, 0.0]);
        ls.initial_step_length(0.8).unwrap();

        let data = ls.backtracking_step(&mut Problem::new(prob), IterState::new());
        assert!(data.is_ok());

        let param = data.as_ref().unwrap().get_param().unwrap();
        let cost = data.as_ref().unwrap().get_cost();
        let gradient = data.as_ref().unwrap().get_grad().unwrap();
        assert_relative_eq!(param[0], 0.6, epsilon = f64::EPSILON);
        assert_relative_eq!(param[1], 0.0, epsilon = f64::EPSILON);
        assert_relative_eq!(cost, 0.6f64.powi(2), epsilon = f64::EPSILON);
        assert_relative_eq!(gradient[0], 2.0 * 0.6, epsilon = f64::EPSILON);
        assert_relative_eq!(gradient[1], 0.0, epsilon = f64::EPSILON);
    }

    #[test]
    fn test_init_armijo() {
        use crate::core::IterState;
        use crate::core::Problem;

        let prob = BTTestProblem {};

        let c: f64 = 0.01;
        let armijo = ArmijoCondition::new(c).unwrap();
        let mut ls: BacktrackingLineSearch<Vec<f64>, Vec<f64>, ArmijoCondition<f64>, f64> =
            BacktrackingLineSearch::new(armijo);

        ls.init_param = Some(vec![-1.0, 0.0]);
        ls.init_cost = f64::infinity();
        // in contrast to the step tests above, it is not necessary to set the init_grad here
        // because it will be computed in init if not present.
        ls.initial_step_length(0.8).unwrap();

        assert_error!(
            ls.init(
                &mut Problem::new(prob.clone()),
                IterState::new().param(ls.init_param.clone().unwrap())
            ),
            ArgminError,
            "Not initialized: \"BacktrackingLineSearch: search_direction must be set.\""
        );

        ls.search_direction(vec![2.0f64, 0.0]);

        let data = ls.init(
            &mut Problem::new(prob),
            IterState::new().param(ls.init_param.clone().unwrap()),
        );
        assert!(data.is_ok());

        let data = data.unwrap().0;

        let param = data.get_param().unwrap();
        let cost = data.get_cost();
        assert_relative_eq!(param[0], 0.6, epsilon = f64::EPSILON);
        assert_relative_eq!(param[1], 0.0, epsilon = f64::EPSILON);
        assert_relative_eq!(cost, 0.6f64.powi(2), epsilon = f64::EPSILON);

        assert!(data.get_grad().is_none());
    }

    #[test]
    fn test_init_wolfe() {
        use crate::core::IterState;
        use crate::core::Problem;

        let prob = BTTestProblem {};

        let c1: f64 = 0.01;
        let c2: f64 = 0.9;
        let wolfe = WolfeCondition::new(c1, c2).unwrap();
        let mut ls: BacktrackingLineSearch<Vec<f64>, Vec<f64>, WolfeCondition<f64>, f64> =
            BacktrackingLineSearch::new(wolfe);

        ls.init_param = Some(vec![-1.0, 0.0]);
        ls.init_cost = f64::infinity();
        // in contrast to the step tests above, it is not necessary to set the init_grad here
        // because it will be computed in init if not present.
        ls.initial_step_length(0.8).unwrap();

        assert_error!(
            ls.init(
                &mut Problem::new(prob.clone()),
                IterState::new().param(ls.init_param.clone().unwrap())
            ),
            ArgminError,
            "Not initialized: \"BacktrackingLineSearch: search_direction must be set.\""
        );

        ls.search_direction(vec![2.0f64, 0.0]);

        let data = ls.init(
            &mut Problem::new(prob),
            IterState::new().param(ls.init_param.clone().unwrap()),
        );
        assert!(data.is_ok());

        let data = data.unwrap().0;

        let param = data.get_param().unwrap();
        let cost = data.get_cost();
        let gradient = data.get_grad().unwrap();
        assert_relative_eq!(param[0], 0.6, epsilon = f64::EPSILON);
        assert_relative_eq!(param[1], 0.0, epsilon = f64::EPSILON);
        assert_relative_eq!(cost, 0.6f64.powi(2), epsilon = f64::EPSILON);
        assert_relative_eq!(gradient[0], 2.0 * 0.6, epsilon = f64::EPSILON);
        assert_relative_eq!(gradient[1], 0.0, epsilon = f64::EPSILON);
    }

    #[test]
    fn test_next_iter() {
        // Similar to step test, but with the added check that self.alpha is reduced.
        use crate::core::Problem;

        let prob = BTTestProblem {};

        let c: f64 = 0.01;
        let armijo = ArmijoCondition::new(c).unwrap();
        let mut ls: BacktrackingLineSearch<Vec<f64>, Vec<f64>, ArmijoCondition<f64>, f64> =
            BacktrackingLineSearch::new(armijo);

        let init_alpha = 0.8;
        ls.init_param = Some(vec![-1.0, 0.0]);
        ls.init_cost = f64::infinity();
        ls.init_grad = Some(vec![-2.0, 0.0]);
        ls.search_direction(vec![2.0f64, 0.0]);
        ls.initial_step_length(init_alpha).unwrap();

        let data = ls.next_iter(
            &mut Problem::new(prob),
            IterState::new().param(ls.init_param.clone().unwrap()),
        );
        assert!(data.is_ok());

        let param = data.as_ref().unwrap().0.get_param().unwrap();
        let cost = data.as_ref().unwrap().0.get_cost();
        // step is smaller than compared to step test, because of the reduced alpha.
        assert_relative_eq!(param[0], 0.44, epsilon = f64::EPSILON);
        assert_relative_eq!(param[1], 0.0, epsilon = f64::EPSILON);
        assert_relative_eq!(cost, 0.44f64.powi(2), epsilon = f64::EPSILON);

        assert!(data.as_ref().unwrap().0.get_grad().is_none());
        assert_relative_eq!(ls.alpha, ls.rho * 0.8, epsilon = f64::EPSILON);
    }

    #[test]
    fn test_termination() {
        let c: f64 = 0.01;
        let armijo = ArmijoCondition::new(c).unwrap();
        let mut ls: BacktrackingLineSearch<Vec<f64>, Vec<f64>, ArmijoCondition<f64>, f64> =
            BacktrackingLineSearch::new(armijo);

        let init_alpha = 0.8;
        ls.init_param = Some(vec![-1.0, 0.0]);
        ls.init_cost = f64::infinity();
        ls.init_grad = Some(vec![-2.0, 0.0]);
        ls.search_direction(vec![2.0f64, 0.0]);
        ls.initial_step_length(init_alpha).unwrap();

        let init_param = ls.init_param.clone().unwrap();
        assert_eq!(
            <BacktrackingLineSearch<Vec<f64>, Vec<f64>, ArmijoCondition<f64>, f64> as Solver<
                TestProblem,
                IterState<Vec<f64>, Vec<f64>, (), (), f64>,
            >>::terminate(
                &mut ls,
                &IterState::<Vec<f64>, Vec<f64>, (), (), f64>::new().param(init_param)
            ),
            TerminationReason::LineSearchConditionMet
        );

        ls.init_cost = 0.0f64;

        let init_param = ls.init_param.clone().unwrap();
        assert_eq!(
            <BacktrackingLineSearch<Vec<f64>, Vec<f64>, ArmijoCondition<f64>, f64> as Solver<
                TestProblem,
                IterState<Vec<f64>, Vec<f64>, (), (), f64>,
            >>::terminate(
                &mut ls,
                &IterState::<Vec<f64>, Vec<f64>, (), (), f64>::new().param(init_param)
            ),
            TerminationReason::NotTerminated
        );
    }

    #[test]
    fn test_executor_1() {
        let prob = BTTestProblem {};

        let c: f64 = 0.01;
        let armijo = ArmijoCondition::new(c).unwrap();
        let mut ls: BacktrackingLineSearch<Vec<f64>, Vec<f64>, ArmijoCondition<f64>, f64> =
            BacktrackingLineSearch::new(armijo);

        ls.init_param = Some(vec![-1.0, 0.0]);
        ls.init_cost = f64::infinity();
        // in contrast to the step tests above, it is not necessary to set the init_grad here
        // because it will be computed in init if not present.
        ls.initial_step_length(0.8).unwrap();

        assert_error!(
            Executor::new(prob.clone(), ls.clone())
                .configure(|config| config.param(ls.init_param.clone().unwrap()).max_iters(10))
                .run(),
            ArgminError,
            "Not initialized: \"BacktrackingLineSearch: search_direction must be set.\""
        );

        ls.search_direction(vec![2.0f64, 0.0]);

        let data = Executor::new(prob, ls.clone())
            .configure(|config| config.param(ls.init_param.clone().unwrap()).max_iters(10))
            .run();
        assert!(data.is_ok());

        let data = data.unwrap().state;

        let param = data.get_param().unwrap();
        assert_relative_eq!(param[0], 0.6, epsilon = f64::EPSILON);
        assert_relative_eq!(param[1], 0.0, epsilon = f64::EPSILON);
        assert_relative_eq!(data.get_cost(), 0.6.powi(2), epsilon = f64::EPSILON);
        assert_eq!(data.iter, 0);
        let func_counts = data.get_func_counts();
        assert_eq!(func_counts["cost_count"], 2);
        assert_eq!(func_counts["gradient_count"], 1);
        assert_eq!(
            data.termination_reason,
            TerminationReason::LineSearchConditionMet
        );

        assert!(data.get_grad().is_none());
    }

    #[test]
    fn test_executor_2() {
        let prob = BTTestProblem {};

        // difference compared to test_executor_1: c is larger to force another backtracking step
        let c: f64 = 0.2;
        let armijo = ArmijoCondition::new(c).unwrap();
        let mut ls: BacktrackingLineSearch<Vec<f64>, Vec<f64>, ArmijoCondition<f64>, f64> =
            BacktrackingLineSearch::new(armijo);

        ls.init_param = Some(vec![-1.0, 0.0]);
        ls.init_cost = f64::infinity();
        // in contrast to the step tests above, it is not necessary to set the init_grad here
        // because it will be computed in init if not present.
        ls.initial_step_length(0.8).unwrap();

        assert_error!(
            Executor::new(prob.clone(), ls.clone())
                .configure(|config| config.param(ls.init_param.clone().unwrap()).max_iters(10))
                .run(),
            ArgminError,
            "Not initialized: \"BacktrackingLineSearch: search_direction must be set.\""
        );

        ls.search_direction(vec![2.0f64, 0.0]);

        let data = Executor::new(prob, ls.clone())
            .configure(|config| config.param(ls.init_param.clone().unwrap()).max_iters(10))
            .run();
        assert!(data.is_ok());

        let data = data.unwrap().state;

        let param = data.get_param().unwrap();
        assert_relative_eq!(param[0], 0.44, epsilon = f64::EPSILON);
        assert_relative_eq!(param[1], 0.0, epsilon = f64::EPSILON);
        assert_relative_eq!(data.get_cost(), 0.44f64.powi(2), epsilon = f64::EPSILON);
        assert_eq!(data.iter, 1);
        let func_counts = data.get_func_counts();
        assert_eq!(func_counts["cost_count"], 3);
        assert_eq!(func_counts["gradient_count"], 1);
        assert_eq!(
            data.termination_reason,
            TerminationReason::LineSearchConditionMet
        );
        assert!(data.get_grad().is_none());
    }
}
