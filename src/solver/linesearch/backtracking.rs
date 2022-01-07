// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! * [Backtracking line search](struct.BacktrackingLineSearch.html)

use crate::prelude::*;
use crate::solver::linesearch::condition::*;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

/// The Backtracking line search is a simple method to find a step length which obeys the Armijo
/// (sufficient decrease) condition.
///
/// # References:
///
/// \[0\] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
///
/// \[1\] Wikipedia: <https://en.wikipedia.org/wiki/Backtracking_line_search>
#[derive(Serialize, Deserialize, Clone)]
pub struct BacktrackingLineSearch<P, L, F> {
    /// initial parameter vector
    init_param: P,
    /// initial cost
    init_cost: F,
    /// initial gradient
    init_grad: P,
    /// Search direction
    search_direction: Option<P>,
    /// Contraction factor rho
    rho: F,
    /// Stopping condition
    condition: L,
    /// alpha
    alpha: F,
}

impl<P: Default, L, F: ArgminFloat> BacktrackingLineSearch<P, L, F> {
    /// Constructor
    pub fn new(condition: L) -> Self {
        BacktrackingLineSearch {
            init_param: P::default(),
            init_cost: F::infinity(),
            init_grad: P::default(),
            search_direction: None,
            rho: F::from_f64(0.9).unwrap(),
            condition,
            alpha: F::from_f64(1.0).unwrap(),
        }
    }

    /// Set rho
    pub fn rho(mut self, rho: F) -> Result<Self, Error> {
        if rho <= F::from_f64(0.0).unwrap() || rho >= F::from_f64(1.0).unwrap() {
            return Err(ArgminError::InvalidParameter {
                text: "BacktrackingLineSearch: Contraction factor rho must be in (0, 1)."
                    .to_string(),
            }
            .into());
        }
        self.rho = rho;
        Ok(self)
    }
}

impl<P, L, F> ArgminLineSearch<P, F> for BacktrackingLineSearch<P, L, F>
where
    P: Clone + Serialize + ArgminSub<P, P> + ArgminDot<P, f64> + ArgminScaledAdd<P, f64, P>,
    L: LineSearchCondition<P, F>,
    F: ArgminFloat + Serialize + DeserializeOwned,
{
    /// Set search direction
    fn set_search_direction(&mut self, search_direction: P) {
        self.search_direction = Some(search_direction);
    }

    /// Set initial alpha value
    fn set_init_alpha(&mut self, alpha: F) -> Result<(), Error> {
        if alpha <= F::from_f64(0.0).unwrap() {
            return Err(ArgminError::InvalidParameter {
                text: "LineSearch: Inital alpha must be > 0.".to_string(),
            }
            .into());
        }
        self.alpha = alpha;
        Ok(())
    }
}

impl<P, L, F: ArgminFloat> BacktrackingLineSearch<P, L, F>
where
    P: ArgminScaledAdd<P, F, P>,
    L: LineSearchCondition<P, F>,
{
    fn backtracking_step<O: ArgminOp<Param = P, Output = F, Float = F>>(
        &self,
        op: &mut OpWrapper<O>,
    ) -> Result<ArgminIterData<O>, Error> {
        let new_param = self
            .init_param
            .scaled_add(&self.alpha, self.search_direction.as_ref().unwrap());

        let cur_cost = op.apply(&new_param)?;

        let out = if self.condition.requires_cur_grad() {
            ArgminIterData::new()
                .grad(op.gradient(&new_param)?)
                .param(new_param)
                .cost(cur_cost)
        } else {
            ArgminIterData::new().param(new_param).cost(cur_cost)
        };

        Ok(out)
    }
}

impl<O, P, L, F> Solver<O> for BacktrackingLineSearch<P, L, F>
where
    P: Clone + Default + Serialize + DeserializeOwned + ArgminScaledAdd<P, F, P>,
    O: ArgminOp<Param = P, Output = F, Float = F>,
    L: LineSearchCondition<P, F>,
    F: ArgminFloat,
{
    const NAME: &'static str = "Backtracking Line search";

    fn init(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &IterState<O>,
    ) -> Result<Option<ArgminIterData<O>>, Error> {
        self.init_param = state.get_param();
        let cost = state.get_cost();
        self.init_cost = if cost == F::infinity() {
            op.apply(&self.init_param)?
        } else {
            cost
        };

        self.init_grad = state
            .get_grad()
            .map(Result::Ok)
            .unwrap_or_else(|| op.gradient(&self.init_param))?;

        if self.search_direction.is_none() {
            return Err(ArgminError::NotInitialized {
                text: "BacktrackingLineSearch: search_direction must be set.".to_string(),
            }
            .into());
        }

        let out = self.backtracking_step(op)?;
        Ok(Some(out))
    }

    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        _state: &IterState<O>,
    ) -> Result<ArgminIterData<O>, Error> {
        self.alpha = self.alpha * self.rho;
        self.backtracking_step(op)
    }

    fn terminate(&mut self, state: &IterState<O>) -> TerminationReason {
        if self.condition.eval(
            state.get_cost(),
            &state
                .get_grad()
                .as_ref()
                .map(Cow::Borrowed)
                .unwrap_or_else(|| Cow::Owned(P::default())),
            self.init_cost,
            &self.init_grad,
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
    use crate::core::MinimalNoOperator;
    use crate::test_trait_impl;
    use approx::assert_relative_eq;

    #[derive(Debug, Clone)]
    struct Problem {}

    impl ArgminOp for Problem {
        type Param = Vec<Self::Float>;
        type Output = Self::Float;
        type Hessian = ();
        type Jacobian = ();
        type Float = f64;

        fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
            Ok(p[0].powi(2) + p[1].powi(2))
        }

        fn gradient(&self, p: &Self::Param) -> Result<Self::Param, Error> {
            Ok(vec![2.0 * p[0], 2.0 * p[1]])
        }
    }

    test_trait_impl!(backtrackinglinesearch,
                    BacktrackingLineSearch<MinimalNoOperator, ArmijoCondition<f64>, f64>);

    #[test]
    fn test_new() {
        let c: f64 = 0.01;
        let armijo = ArmijoCondition::new(c).unwrap();
        let ls: BacktrackingLineSearch<Vec<f64>, ArmijoCondition<f64>, f64> =
            BacktrackingLineSearch::new(armijo);

        assert_eq!(ls.init_param, Vec::<f64>::default());
        assert!(ls.init_cost.is_infinite());
        assert!(ls.init_cost.is_sign_positive());
        assert_eq!(ls.init_grad, Vec::<f64>::default());
        assert_eq!(ls.search_direction, None);
        assert_eq!(ls.rho.to_ne_bytes(), 0.9f64.to_ne_bytes());
        assert_eq!(ls.alpha.to_ne_bytes(), 1.0f64.to_ne_bytes());
    }

    #[test]
    fn test_rho() {
        let c: f64 = 0.01;
        let armijo = ArmijoCondition::new(c).unwrap();
        let ls: BacktrackingLineSearch<Vec<f64>, ArmijoCondition<f64>, f64> =
            BacktrackingLineSearch::new(armijo);

        assert_error!(
            ls.rho(1.0f64),
            ArgminError,
            "Invalid parameter: \"BacktrackingLineSearch: Contraction factor rho must be in (0, 1).\""
        );

        let c: f64 = 0.01;
        let armijo = ArmijoCondition::new(c).unwrap();
        let ls: BacktrackingLineSearch<Vec<f64>, ArmijoCondition<f64>, f64> =
            BacktrackingLineSearch::new(armijo);

        assert_error!(
            ls.rho(0.0f64),
            ArgminError,
            "Invalid parameter: \"BacktrackingLineSearch: Contraction factor rho must be in (0, 1).\""
        );

        let c: f64 = 0.01;
        let armijo = ArmijoCondition::new(c).unwrap();
        let ls: BacktrackingLineSearch<Vec<f64>, ArmijoCondition<f64>, f64> =
            BacktrackingLineSearch::new(armijo);

        assert!(ls.rho(0.0f64 + f64::EPSILON).is_ok());

        let c: f64 = 0.01;
        let armijo = ArmijoCondition::new(c).unwrap();
        let ls: BacktrackingLineSearch<Vec<f64>, ArmijoCondition<f64>, f64> =
            BacktrackingLineSearch::new(armijo);

        assert!(ls.rho(1.0f64 - f64::EPSILON).is_ok());
    }

    #[test]
    fn test_set_search_direction() {
        let c: f64 = 0.01;
        let armijo = ArmijoCondition::new(c).unwrap();
        let mut ls: BacktrackingLineSearch<Vec<f64>, ArmijoCondition<f64>, f64> =
            BacktrackingLineSearch::new(armijo);
        ls.set_search_direction(vec![1.0f64, 1.0]);

        assert_eq!(ls.search_direction, Some(vec![1.0f64, 1.0]));
    }

    #[test]
    fn test_set_init_alpha() {
        let c: f64 = 0.01;
        let armijo = ArmijoCondition::new(c).unwrap();
        let mut ls: BacktrackingLineSearch<Vec<f64>, ArmijoCondition<f64>, f64> =
            BacktrackingLineSearch::new(armijo);

        assert!(ls.set_init_alpha(f64::EPSILON).is_ok());

        assert_error!(
            ls.set_init_alpha(0.0f64),
            ArgminError,
            "Invalid parameter: \"LineSearch: Inital alpha must be > 0.\""
        );
    }

    #[test]
    fn test_step_armijo() {
        use crate::core::OpWrapper;

        let prob = Problem {};

        let c: f64 = 0.01;
        let armijo = ArmijoCondition::new(c).unwrap();
        let mut ls: BacktrackingLineSearch<Vec<f64>, ArmijoCondition<f64>, f64> =
            BacktrackingLineSearch::new(armijo);

        ls.init_param = vec![-1.0, 0.0];
        ls.init_cost = f64::infinity();
        ls.init_grad = vec![-2.0, 0.0];
        ls.set_search_direction(vec![2.0f64, 0.0]);
        ls.set_init_alpha(0.8).unwrap();

        let data = ls.backtracking_step(&mut OpWrapper::new(prob));
        assert!(data.is_ok());

        let param = data.as_ref().unwrap().get_param().unwrap();
        let cost = data.as_ref().unwrap().get_cost().unwrap();
        assert_relative_eq!(param[0], 0.6, epsilon = f64::EPSILON);
        assert_relative_eq!(param[1], 0.0, epsilon = f64::EPSILON);
        assert_relative_eq!(cost, 0.6.powi(2), epsilon = f64::EPSILON);

        assert!(data.as_ref().unwrap().get_grad().is_none());
    }

    #[test]
    fn test_step_wolfe() {
        // Wolfe, in contrast to Armijo, requires the current gradient. This test makes sure that
        // the implementation of the backtracking linesearch properly considers this.
        use crate::core::OpWrapper;

        let prob = Problem {};

        let c1: f64 = 0.01;
        let c2: f64 = 0.9;
        let wolfe = WolfeCondition::new(c1, c2).unwrap();
        let mut ls: BacktrackingLineSearch<Vec<f64>, WolfeCondition<f64>, f64> =
            BacktrackingLineSearch::new(wolfe);

        ls.init_param = vec![-1.0, 0.0];
        ls.init_cost = f64::infinity();
        ls.init_grad = vec![-2.0, 0.0];
        ls.set_search_direction(vec![2.0f64, 0.0]);
        ls.set_init_alpha(0.8).unwrap();

        let data = ls.backtracking_step(&mut OpWrapper::new(prob));
        assert!(data.is_ok());

        let param = data.as_ref().unwrap().get_param().unwrap();
        let cost = data.as_ref().unwrap().get_cost().unwrap();
        let gradient = data.as_ref().unwrap().get_grad().unwrap();
        assert_relative_eq!(param[0], 0.6, epsilon = f64::EPSILON);
        assert_relative_eq!(param[1], 0.0, epsilon = f64::EPSILON);
        assert_relative_eq!(cost, 0.6.powi(2), epsilon = f64::EPSILON);
        assert_relative_eq!(gradient[0], 2.0 * 0.6, epsilon = f64::EPSILON);
        assert_relative_eq!(gradient[1], 0.0, epsilon = f64::EPSILON);
    }

    #[test]
    fn test_init_armijo() {
        use crate::core::IterState;
        use crate::core::OpWrapper;

        let prob = Problem {};

        let c: f64 = 0.01;
        let armijo = ArmijoCondition::new(c).unwrap();
        let mut ls: BacktrackingLineSearch<Vec<f64>, ArmijoCondition<f64>, f64> =
            BacktrackingLineSearch::new(armijo);

        ls.init_param = vec![-1.0, 0.0];
        ls.init_cost = f64::infinity();
        // in contrast to the step tests above, it is not necessary to set the init_grad here
        // because it will be computed in init if not present.
        ls.set_init_alpha(0.8).unwrap();

        assert_error!(
            ls.init(
                &mut OpWrapper::new(prob.clone()),
                &IterState::new(ls.init_param.clone())
            ),
            ArgminError,
            "Not initialized: \"BacktrackingLineSearch: search_direction must be set.\""
        );

        ls.set_search_direction(vec![2.0f64, 0.0]);

        let data = ls.init(
            &mut OpWrapper::new(prob),
            &IterState::new(ls.init_param.clone()),
        );
        assert!(data.is_ok());

        let data = data.as_ref().unwrap().as_ref().unwrap();

        let param = data.get_param().unwrap();
        let cost = data.get_cost().unwrap();
        assert_relative_eq!(param[0], 0.6, epsilon = f64::EPSILON);
        assert_relative_eq!(param[1], 0.0, epsilon = f64::EPSILON);
        assert_relative_eq!(cost, 0.6.powi(2), epsilon = f64::EPSILON);

        assert!(data.get_grad().is_none());
    }

    #[test]
    fn test_init_wolfe() {
        use crate::core::IterState;
        use crate::core::OpWrapper;

        let prob = Problem {};

        let c1: f64 = 0.01;
        let c2: f64 = 0.9;
        let wolfe = WolfeCondition::new(c1, c2).unwrap();
        let mut ls: BacktrackingLineSearch<Vec<f64>, WolfeCondition<f64>, f64> =
            BacktrackingLineSearch::new(wolfe);

        ls.init_param = vec![-1.0, 0.0];
        ls.init_cost = f64::infinity();
        // in contrast to the step tests above, it is not necessary to set the init_grad here
        // because it will be computed in init if not present.
        ls.set_init_alpha(0.8).unwrap();

        assert_error!(
            ls.init(
                &mut OpWrapper::new(prob.clone()),
                &IterState::new(ls.init_param.clone())
            ),
            ArgminError,
            "Not initialized: \"BacktrackingLineSearch: search_direction must be set.\""
        );

        ls.set_search_direction(vec![2.0f64, 0.0]);

        let data = ls.init(
            &mut OpWrapper::new(prob),
            &IterState::new(ls.init_param.clone()),
        );
        assert!(data.is_ok());

        let data = data.as_ref().unwrap().as_ref().unwrap();

        let param = data.get_param().unwrap();
        let cost = data.get_cost().unwrap();
        let gradient = data.get_grad().unwrap();
        assert_relative_eq!(param[0], 0.6, epsilon = f64::EPSILON);
        assert_relative_eq!(param[1], 0.0, epsilon = f64::EPSILON);
        assert_relative_eq!(cost, 0.6.powi(2), epsilon = f64::EPSILON);
        assert_relative_eq!(gradient[0], 2.0 * 0.6, epsilon = f64::EPSILON);
        assert_relative_eq!(gradient[1], 0.0, epsilon = f64::EPSILON);
    }

    #[test]
    fn test_next_iter() {
        // Similar to step test, but with the added check that self.alpha is reduced.
        use crate::core::OpWrapper;

        let prob = Problem {};

        let c: f64 = 0.01;
        let armijo = ArmijoCondition::new(c).unwrap();
        let mut ls: BacktrackingLineSearch<Vec<f64>, ArmijoCondition<f64>, f64> =
            BacktrackingLineSearch::new(armijo);

        let init_alpha = 0.8;
        ls.init_param = vec![-1.0, 0.0];
        ls.init_cost = f64::infinity();
        ls.init_grad = vec![-2.0, 0.0];
        ls.set_search_direction(vec![2.0f64, 0.0]);
        ls.set_init_alpha(init_alpha).unwrap();

        let data = ls.next_iter(
            &mut OpWrapper::new(prob),
            &IterState::new(ls.init_param.clone()),
        );
        assert!(data.is_ok());

        let param = data.as_ref().unwrap().get_param().unwrap();
        let cost = data.as_ref().unwrap().get_cost().unwrap();
        // step is smaller than compared to step test, because of the reduced alpha.
        assert_relative_eq!(param[0], 0.44, epsilon = f64::EPSILON);
        assert_relative_eq!(param[1], 0.0, epsilon = f64::EPSILON);
        assert_relative_eq!(cost, 0.44.powi(2), epsilon = f64::EPSILON);

        assert!(data.as_ref().unwrap().get_grad().is_none());
        assert_relative_eq!(ls.alpha, ls.rho * 0.8, epsilon = f64::EPSILON);
    }

    #[test]
    fn test_termination() {
        let c: f64 = 0.01;
        let armijo = ArmijoCondition::new(c).unwrap();
        let mut ls: BacktrackingLineSearch<Vec<f64>, ArmijoCondition<f64>, f64> =
            BacktrackingLineSearch::new(armijo);

        let init_alpha = 0.8;
        ls.init_param = vec![-1.0, 0.0];
        ls.init_cost = f64::infinity();
        ls.init_grad = vec![-2.0, 0.0];
        ls.set_search_direction(vec![2.0f64, 0.0]);
        ls.set_init_alpha(init_alpha).unwrap();

        assert_eq!(
            ls.terminate(&IterState::<Problem>::new(ls.init_param.clone())),
            TerminationReason::LineSearchConditionMet
        );

        ls.init_cost = 0.0f64;

        assert_eq!(
            ls.terminate(&IterState::<Problem>::new(ls.init_param.clone())),
            TerminationReason::NotTerminated
        );
    }

    #[test]
    fn test_executor_1() {
        let prob = Problem {};

        let c: f64 = 0.01;
        let armijo = ArmijoCondition::new(c).unwrap();
        let mut ls: BacktrackingLineSearch<Vec<f64>, ArmijoCondition<f64>, f64> =
            BacktrackingLineSearch::new(armijo);

        ls.init_param = vec![-1.0, 0.0];
        ls.init_cost = f64::infinity();
        // in contrast to the step tests above, it is not necessary to set the init_grad here
        // because it will be computed in init if not present.
        ls.set_init_alpha(0.8).unwrap();

        assert_error!(
            Executor::new(prob.clone(), ls.clone(), ls.init_param.clone())
                .max_iters(10)
                .run(),
            ArgminError,
            "Not initialized: \"BacktrackingLineSearch: search_direction must be set.\""
        );

        ls.set_search_direction(vec![2.0f64, 0.0]);

        let data = Executor::new(prob, ls.clone(), ls.init_param.clone())
            .max_iters(10)
            .run();
        assert!(data.is_ok());

        let data = data.unwrap().state;

        let param = data.get_param();
        assert_relative_eq!(param[0], 0.6, epsilon = f64::EPSILON);
        assert_relative_eq!(param[1], 0.0, epsilon = f64::EPSILON);
        assert_relative_eq!(data.get_cost(), 0.6.powi(2), epsilon = f64::EPSILON);
        assert_eq!(data.iter, 0);
        assert_eq!(data.cost_func_count, 2);
        assert_eq!(data.grad_func_count, 1);
        assert_eq!(data.hessian_func_count, 0);
        assert_eq!(data.jacobian_func_count, 0);
        assert_eq!(data.modify_func_count, 0);
        assert_eq!(
            data.termination_reason,
            TerminationReason::LineSearchConditionMet
        );

        assert!(data.get_grad().is_none());
    }

    #[test]
    fn test_executor_2() {
        let prob = Problem {};

        // difference compared to test_executor_1: c is larger to force another backtracking step
        let c: f64 = 0.2;
        let armijo = ArmijoCondition::new(c).unwrap();
        let mut ls: BacktrackingLineSearch<Vec<f64>, ArmijoCondition<f64>, f64> =
            BacktrackingLineSearch::new(armijo);

        ls.init_param = vec![-1.0, 0.0];
        ls.init_cost = f64::infinity();
        // in contrast to the step tests above, it is not necessary to set the init_grad here
        // because it will be computed in init if not present.
        ls.set_init_alpha(0.8).unwrap();

        assert_error!(
            Executor::new(prob.clone(), ls.clone(), ls.init_param.clone())
                .max_iters(10)
                .run(),
            ArgminError,
            "Not initialized: \"BacktrackingLineSearch: search_direction must be set.\""
        );

        ls.set_search_direction(vec![2.0f64, 0.0]);

        let data = Executor::new(prob, ls.clone(), ls.init_param.clone())
            .max_iters(10)
            .run();
        assert!(data.is_ok());

        let data = data.unwrap().state;

        let param = data.get_param();
        assert_relative_eq!(param[0], 0.44, epsilon = f64::EPSILON);
        assert_relative_eq!(param[1], 0.0, epsilon = f64::EPSILON);
        assert_relative_eq!(data.get_cost(), 0.44.powi(2), epsilon = f64::EPSILON);
        assert_eq!(data.iter, 1);
        assert_eq!(data.cost_func_count, 3);
        assert_eq!(data.grad_func_count, 1);
        assert_eq!(data.hessian_func_count, 0);
        assert_eq!(data.jacobian_func_count, 0);
        assert_eq!(data.modify_func_count, 0);
        assert_eq!(
            data.termination_reason,
            TerminationReason::LineSearchConditionMet
        );

        assert!(data.get_grad().is_none());
    }
}
