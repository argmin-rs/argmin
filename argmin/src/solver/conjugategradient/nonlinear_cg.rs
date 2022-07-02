// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::{
    ArgminFloat, CostFunction, DeserializeOwnedAlias, Error, Executor, Gradient, IterState,
    LineSearch, NLCGBetaUpdate, OptimizationResult, Problem, SerializeAlias, Solver, State, KV,
};
use argmin_math::{ArgminAdd, ArgminDot, ArgminMul, ArgminNorm};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// # Non-linear Conjugate Gradient method
///
/// A generalization of the conjugate gradient method for nonlinear optimization problems.
///
/// Requires an initial parameter vector.
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
pub struct NonlinearConjugateGradient<P, L, B, F> {
    /// p
    p: Option<P>,
    /// beta
    beta: F,
    /// line search
    linesearch: L,
    /// beta update method
    beta_method: B,
    /// Number of iterations after which a restart is performed
    restart_iter: u64,
    /// Restart based on orthogonality
    restart_orthogonality: Option<F>,
}

impl<P, L, B, F> NonlinearConjugateGradient<P, L, B, F>
where
    F: ArgminFloat,
{
    /// Construct a new instance of `NonlinearConjugateGradient`.
    ///
    /// Takes a [`LineSearch`] and a [`NLCGBetaUpdate`] as input.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::conjugategradient::NonlinearConjugateGradient;
    /// # let linesearch = ();
    /// # let beta_method = ();
    /// let nlcg: NonlinearConjugateGradient<Vec<f64>, _, _, f64> =
    ///     NonlinearConjugateGradient::new(linesearch, beta_method);
    /// ```
    pub fn new(linesearch: L, beta_method: B) -> Self {
        NonlinearConjugateGradient {
            p: None,
            beta: F::nan(),
            linesearch,
            beta_method,
            restart_iter: std::u64::MAX,
            restart_orthogonality: None,
        }
    }

    /// Specifiy the number of iterations after which a restart should be performed.
    ///
    /// This allows the algorithm to "forget" previous information which may not be helpful
    /// anymore.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::conjugategradient::NonlinearConjugateGradient;
    /// # let linesearch = ();
    /// # let beta_method = ();
    /// # let nlcg: NonlinearConjugateGradient<Vec<f64>, _, _, f64> = NonlinearConjugateGradient::new(linesearch, beta_method);
    /// let nlcg = nlcg.restart_iters(100);
    /// ```
    #[must_use]
    pub fn restart_iters(mut self, iters: u64) -> Self {
        self.restart_iter = iters;
        self
    }

    /// Set the value for the orthogonality measure.
    ///
    /// Setting this parameter leads to a restart of the algorithm (setting beta = 0) after
    /// consecutive search directions stop being orthogonal. In other words, if this condition
    /// is met:
    ///
    /// `|\nabla f_k^T * \nabla f_{k-1}| / | \nabla f_k |^2 >= v`
    ///
    /// A typical value for `v` is 0.1.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::conjugategradient::NonlinearConjugateGradient;
    /// # let linesearch = ();
    /// # let beta_method = ();
    /// # let nlcg: NonlinearConjugateGradient<Vec<f64>, _, _, f64> = NonlinearConjugateGradient::new(linesearch, beta_method);
    /// let nlcg = nlcg.restart_orthogonality(0.1);
    /// ```
    #[must_use]
    pub fn restart_orthogonality(mut self, v: F) -> Self {
        self.restart_orthogonality = Some(v);
        self
    }
}

impl<O, P, G, L, B, F> Solver<O, IterState<P, G, (), (), F>>
    for NonlinearConjugateGradient<P, L, B, F>
where
    O: CostFunction<Param = P, Output = F> + Gradient<Param = P, Gradient = G>,
    P: Clone + SerializeAlias + DeserializeOwnedAlias + ArgminAdd<P, P> + ArgminMul<F, P>,
    G: Clone
        + SerializeAlias
        + DeserializeOwnedAlias
        + ArgminMul<F, P>
        + ArgminDot<G, F>
        + ArgminNorm<F>,
    L: Clone + LineSearch<P, F> + Solver<O, IterState<P, G, (), (), F>>,
    B: NLCGBetaUpdate<G, P, F>,
    F: ArgminFloat,
{
    const NAME: &'static str = "Nonlinear Conjugate Gradient";

    fn init(
        &mut self,
        problem: &mut Problem<O>,
        state: IterState<P, G, (), (), F>,
    ) -> Result<(IterState<P, G, (), (), F>, Option<KV>), Error> {
        let param = state.get_param().ok_or_else(argmin_error_closure!(
            NotInitialized,
            concat!(
                "`NonlinearConjugateGradient` requires an initial parameter vector. ",
                "Please provide an initial guess via `Executor`s `configure` method."
            )
        ))?;
        let cost = problem.cost(param)?;
        let grad = problem.gradient(param)?;
        self.p = Some(grad.mul(&(float!(-1.0))));
        Ok((state.cost(cost).grad(grad), None))
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        mut state: IterState<P, G, (), (), F>,
    ) -> Result<(IterState<P, G, (), (), F>, Option<KV>), Error> {
        let p = self.p.as_ref().ok_or_else(argmin_error_closure!(
            PotentialBug,
            "`NonlinearConjugateGradient`: Field `p` not set"
        ))?;
        let xk = state.take_param().ok_or_else(argmin_error_closure!(
            PotentialBug,
            "`NonlinearConjugateGradient`: No `param` in `state`"
        ))?;
        let grad = state
            .take_grad()
            .map(Result::Ok)
            .unwrap_or_else(|| problem.gradient(&xk))?;
        let cur_cost = state.cost;

        // Linesearch
        self.linesearch.search_direction(p.clone());

        // Run solver
        let OptimizationResult {
            problem: line_problem,
            state: mut line_state,
            ..
        } = Executor::new(
            problem.take_problem().ok_or_else(argmin_error_closure!(
                PotentialBug,
                "`NonlinearConjugateGradient`: Failed to take `problem` for line search"
            ))?,
            self.linesearch.clone(),
        )
        .configure(|state| state.param(xk).grad(grad.clone()).cost(cur_cost))
        .ctrlc(false)
        .run()?;

        // takes care of the counts of function evaluations
        problem.consume_problem(line_problem);

        let xk1 = line_state.take_param().ok_or_else(argmin_error_closure!(
            PotentialBug,
            "`NonlinearConjugateGradient`: No `param` returned by line search"
        ))?;

        // Update of beta
        let new_grad = problem.gradient(&xk1)?;

        let restart_orthogonality = match self.restart_orthogonality {
            Some(v) => new_grad.dot(&grad).abs() / new_grad.norm().powi(2) >= v,
            None => false,
        };

        let restart_iter: bool =
            (state.get_iter() % self.restart_iter == 0) && state.get_iter() != 0;

        if restart_iter || restart_orthogonality {
            self.beta = float!(0.0);
        } else {
            self.beta = self.beta_method.update(&grad, &new_grad, p);
        }

        // Update of p
        self.p = Some(new_grad.mul(&(float!(-1.0))).add(&p.mul(&self.beta)));

        // Housekeeping
        let cost = problem.cost(&xk1)?;

        Ok((
            state.param(xk1).cost(cost).grad(new_grad),
            Some(make_kv!("beta" => self.beta;
             "restart_iter" => restart_iter;
             "restart_orthogonality" => restart_orthogonality;
            )),
        ))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::let_unit_value)]

    use super::*;
    use crate::core::test_utils::TestProblem;
    use crate::core::ArgminError;
    use crate::solver::conjugategradient::beta::PolakRibiere;
    use crate::solver::linesearch::{
        condition::ArmijoCondition, BacktrackingLineSearch, MoreThuenteLineSearch,
    };
    use crate::test_trait_impl;
    use approx::assert_relative_eq;

    #[derive(Eq, PartialEq, Clone, Copy, Debug)]
    struct Linesearch {}

    #[derive(Eq, PartialEq, Clone, Copy, Debug)]
    struct BetaUpdate {}

    test_trait_impl!(
        nonlinear_cg,
        NonlinearConjugateGradient<
            TestProblem,
            MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64>,
            PolakRibiere,
            f64
        >
    );

    #[test]
    fn test_new() {
        let linesearch = Linesearch {};
        let beta_method = BetaUpdate {};
        let nlcg: NonlinearConjugateGradient<Vec<f64>, _, _, f64> =
            NonlinearConjugateGradient::new(linesearch, beta_method);
        let NonlinearConjugateGradient {
            p,
            beta,
            linesearch,
            beta_method,
            restart_iter,
            restart_orthogonality,
        } = nlcg;
        assert!(p.is_none());
        assert!(beta.is_nan());
        assert_eq!(linesearch, linesearch);
        assert_eq!(beta_method, beta_method);
        assert_eq!(restart_iter, std::u64::MAX);
        assert!(restart_orthogonality.is_none());
    }

    #[test]
    fn test_restart_iters() {
        let linesearch = ();
        let beta_method = ();
        let nlcg: NonlinearConjugateGradient<Vec<f64>, _, _, f64> =
            NonlinearConjugateGradient::new(linesearch, beta_method);
        assert_eq!(nlcg.restart_iter, std::u64::MAX);
        let nlcg = nlcg.restart_iters(100);
        assert_eq!(nlcg.restart_iter, 100);
    }

    #[test]
    fn test_restart_orthogonality() {
        let linesearch = ();
        let beta_method = ();
        let nlcg: NonlinearConjugateGradient<Vec<f64>, _, _, f64> =
            NonlinearConjugateGradient::new(linesearch, beta_method);
        assert!(nlcg.restart_orthogonality.is_none());
        let nlcg = nlcg.restart_orthogonality(0.1);
        assert_eq!(
            nlcg.restart_orthogonality.as_ref().unwrap().to_ne_bytes(),
            0.1f64.to_ne_bytes()
        );
    }

    #[test]
    fn test_init_param_not_initialized() {
        let linesearch: BacktrackingLineSearch<Vec<f64>, Vec<f64>, ArmijoCondition<f64>, f64> =
            BacktrackingLineSearch::new(ArmijoCondition::new(0.2).unwrap());
        let beta_method = PolakRibiere::new();
        let mut nlcg: NonlinearConjugateGradient<Vec<f64>, _, _, f64> =
            NonlinearConjugateGradient::new(linesearch, beta_method);
        let res = nlcg.init(&mut Problem::new(TestProblem::new()), IterState::new());
        assert_error!(
            res,
            ArgminError,
            concat!(
                "Not initialized: \"`NonlinearConjugateGradient` requires an initial parameter vector. ",
                "Please provide an initial guess via `Executor`s `configure` method.\""
            )
        );
    }

    #[test]
    fn test_init() {
        let linesearch: BacktrackingLineSearch<Vec<f64>, Vec<f64>, ArmijoCondition<f64>, f64> =
            BacktrackingLineSearch::new(ArmijoCondition::new(0.2).unwrap());
        let beta_method = PolakRibiere::new();
        let mut nlcg: NonlinearConjugateGradient<Vec<f64>, _, _, f64> =
            NonlinearConjugateGradient::new(linesearch, beta_method);
        let state: IterState<Vec<f64>, Vec<f64>, (), (), f64> =
            IterState::new().param(vec![3.0, 4.0]);
        let (state_out, kv) = nlcg
            .init(&mut Problem::new(TestProblem::new()), state.clone())
            .unwrap();
        assert!(kv.is_none());
        assert_ne!(state_out, state);
        assert_eq!(state_out.cost.to_ne_bytes(), 1f64.to_ne_bytes());
        assert_eq!(
            state_out.grad.as_ref().unwrap()[0].to_ne_bytes(),
            3f64.to_ne_bytes()
        );
        assert_eq!(
            state_out.grad.as_ref().unwrap()[1].to_ne_bytes(),
            4f64.to_ne_bytes()
        );
        assert_eq!(
            state_out.param.as_ref().unwrap()[0].to_ne_bytes(),
            3f64.to_ne_bytes()
        );
        assert_eq!(
            state_out.param.as_ref().unwrap()[1].to_ne_bytes(),
            4f64.to_ne_bytes()
        );
        assert_eq!(
            nlcg.p.as_ref().unwrap()[0].to_ne_bytes(),
            (-3f64).to_ne_bytes()
        );
        assert_eq!(
            nlcg.p.as_ref().unwrap()[1].to_ne_bytes(),
            (-4f64).to_ne_bytes()
        );
    }

    #[test]
    fn test_next_iter_p_not_set() {
        let linesearch: BacktrackingLineSearch<Vec<f64>, Vec<f64>, ArmijoCondition<f64>, f64> =
            BacktrackingLineSearch::new(ArmijoCondition::new(0.2).unwrap());
        let beta_method = PolakRibiere::new();
        let mut nlcg: NonlinearConjugateGradient<Vec<f64>, _, _, f64> =
            NonlinearConjugateGradient::new(linesearch, beta_method);
        let state = IterState::new().param(vec![1.0f64, 2.0f64]);
        assert!(nlcg.p.is_none());
        let res = nlcg.next_iter(&mut Problem::new(TestProblem::new()), state);
        assert_error!(
            res,
            ArgminError,
            concat!(
                "Potential bug: \"`NonlinearConjugateGradient`: ",
                "Field `p` not set\". This is potentially a bug. ",
                "Please file a report on https://github.com/argmin-rs/argmin/issues"
            )
        );
    }

    #[test]
    fn test_next_iter_state_param_not_set() {
        let linesearch: BacktrackingLineSearch<Vec<f64>, Vec<f64>, ArmijoCondition<f64>, f64> =
            BacktrackingLineSearch::new(ArmijoCondition::new(0.2).unwrap());
        let beta_method = PolakRibiere::new();
        let mut nlcg: NonlinearConjugateGradient<Vec<f64>, _, _, f64> =
            NonlinearConjugateGradient::new(linesearch, beta_method);
        let state = IterState::new();
        nlcg.p = Some(vec![]);
        assert!(nlcg.p.is_some());
        let res = nlcg.next_iter(&mut Problem::new(TestProblem::new()), state);
        assert_error!(
            res,
            ArgminError,
            concat!(
                "Potential bug: \"`NonlinearConjugateGradient`: ",
                "No `param` in `state`\". This is potentially a bug. ",
                "Please file a report on https://github.com/argmin-rs/argmin/issues"
            )
        );
    }

    #[test]
    fn test_next_iter_problem_missing() {
        let linesearch: BacktrackingLineSearch<Vec<f64>, Vec<f64>, ArmijoCondition<f64>, f64> =
            BacktrackingLineSearch::new(ArmijoCondition::new(0.2).unwrap());
        let beta_method = PolakRibiere::new();
        let mut nlcg: NonlinearConjugateGradient<Vec<f64>, _, _, f64> =
            NonlinearConjugateGradient::new(linesearch, beta_method);
        let state = IterState::new()
            .param(vec![1.0f64, 2.0])
            .grad(vec![1.0f64, 2.0]);
        nlcg.p = Some(vec![]);
        assert!(nlcg.p.is_some());
        let mut problem = Problem::new(TestProblem::new());
        let _ = problem.take_problem().unwrap();
        let res = nlcg.next_iter(&mut problem, state);
        assert_error!(
            res,
            ArgminError,
            concat!(
                "Potential bug: \"`NonlinearConjugateGradient`: ",
                "Failed to take `problem` for line search\". This is potentially a bug. ",
                "Please file a report on https://github.com/argmin-rs/argmin/issues"
            )
        );
    }

    #[test]
    fn test_next_iter() {
        let linesearch: BacktrackingLineSearch<Vec<f64>, Vec<f64>, ArmijoCondition<f64>, f64> =
            BacktrackingLineSearch::new(ArmijoCondition::new(0.2).unwrap());
        let beta_method = PolakRibiere::new();
        let mut nlcg: NonlinearConjugateGradient<Vec<f64>, _, _, f64> =
            NonlinearConjugateGradient::new(linesearch, beta_method);
        let state = IterState::new()
            .param(vec![1.0f64, 2.0])
            .grad(vec![1.0f64, 2.0]);
        let mut problem = Problem::new(TestProblem::new());
        let (state, kv) = nlcg.init(&mut problem, state).unwrap();
        assert!(kv.is_none());
        let (mut state, kv) = nlcg.next_iter(&mut problem, state).unwrap();
        state.update();
        let kv2 = make_kv!("beta" => 0; "restart_iter" => false; "restart_orthogonality" => false;);
        for ((k1, v1), (k2, v2)) in kv.unwrap().kv.iter().zip(kv2.kv.iter()) {
            assert_eq!(k1, k2);
            assert_eq!(format!("{}", v1), format!("{}", v2));
        }
        assert_relative_eq!(
            state.param.as_ref().unwrap()[0],
            1.0f64,
            epsilon = f64::EPSILON
        );
        assert_relative_eq!(
            state.param.as_ref().unwrap()[1],
            2.0f64,
            epsilon = f64::EPSILON
        );
        assert_relative_eq!(
            state.best_param.as_ref().unwrap()[0],
            1.0f64,
            epsilon = f64::EPSILON
        );
        assert_relative_eq!(
            state.best_param.as_ref().unwrap()[1],
            2.0f64,
            epsilon = f64::EPSILON
        );
        assert_relative_eq!(state.cost, 1.0f64, epsilon = f64::EPSILON);
        assert_relative_eq!(state.prev_cost, 1.0f64, epsilon = f64::EPSILON);
        assert_relative_eq!(state.best_cost, 1.0f64, epsilon = f64::EPSILON);
        assert_relative_eq!(
            state.grad.as_ref().unwrap()[0],
            1.0f64,
            epsilon = f64::EPSILON
        );
        assert_relative_eq!(
            state.grad.as_ref().unwrap()[1],
            2.0f64,
            epsilon = f64::EPSILON
        );
    }
}
