// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::{
    ArgminFloat, CostFunction, DeserializeOwnedAlias, Error, Executor, Gradient, IterState,
    LineSearch, OptimizationResult, Problem, SerializeAlias, Solver, State, TerminationReason, KV,
};
use argmin_math::{
    ArgminAdd, ArgminDot, ArgminL1Norm, ArgminMinMax, ArgminMul, ArgminNorm, ArgminSignum,
    ArgminSub, ArgminZeroLike,
};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::marker::PhantomData;

/// Calculates pseudo-gradient of OWL-QN method.
fn calculate_pseudo_gradient<P, G, F>(l1_coeff: F, param: &P, gradient: &G) -> G
where
    P: ArgminAdd<F, P> + ArgminSub<F, P> + ArgminMul<F, P> + ArgminSignum,
    G: ArgminAdd<G, G> + ArgminAdd<P, G> + ArgminMinMax + ArgminZeroLike,
    F: ArgminFloat,
{
    let coeff_p = param.add(&F::min_positive_value()).signum().mul(&l1_coeff);
    let coeff_n = param.sub(&F::min_positive_value()).signum().mul(&l1_coeff);
    let zeros = gradient.zero_like();
    G::max(&gradient.add(&coeff_n), &zeros).add(&G::min(&gradient.add(&coeff_p), &zeros))
}

/// # Limited-memory BFGS (L-BFGS) method
///
/// L-BFGS is an approximation to BFGS which requires a limited amount of memory. Instead of
/// storing the inverse, only a few vectors which implicitly represent the inverse matrix are
/// stored.
///
/// It requires a line search and the number of vectors to be stored (history size `m`) must be
/// set. Additionally an initial guess for the parameter vector is required, which is to be
/// provided via the [`configure`](`crate::core::Executor::configure`) method of the
/// [`Executor`](`crate::core::Executor`) (See [`IterState`], in particular [`IterState::param`]).
/// In the same way the initial gradient and cost function corresponding to the initial parameter
/// vector can be provided. If these are not provided, they will be computed during initialization
/// of the algorithm.
///
/// Two tolerances can be configured, which are both needed in the stopping criteria.
/// One is a tolerance on the gradient (set with
/// [`with_tolerance_grad`](`LBFGS::with_tolerance_grad`)): If the norm of the gradient is below
/// said tolerance, the algorithm stops. It defaults to `sqrt(EPSILON)`.
/// The other one is a tolerance on the change of the cost function from one iteration to the
/// other. If the change is below this tolerance (default: `EPSILON`), the algorithm stops. This
/// parameter can be set via [`with_tolerance_cost`](`LBFGS::with_tolerance_cost`).
///
/// TODO: Implement compact representation of BFGS updating (Nocedal/Wright p.230)
///
/// ## Requirements on the optimization problem
///
/// The optimization problem is required to implement [`CostFunction`] and [`Gradient`].
///
/// ## Reference
///
/// Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
///
/// Galen Andrew and Jianfeng Gao (2007). Scalable Training of L1-Regularized Log-Linear Models,
/// International Conference on Machine Learning.
#[derive(Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct LBFGS<L, P, G, F> {
    /// line search
    linesearch: L,
    /// m
    m: usize,
    /// s_{k-1}
    s: VecDeque<P>,
    /// y_{k-1}
    y: VecDeque<G>,
    /// Tolerance for the stopping criterion based on the change of the norm on the gradient
    tol_grad: F,
    /// Tolerance for the stopping criterion based on the change of the cost stopping criterion
    tol_cost: F,
    /// Coefficient of L1-regularization
    l1_coeff: Option<F>,
    /// Unregularized gradient used for calculation of `y`.
    l1_prev_unreg_grad: Option<G>,
}

impl<L, P, G, F> LBFGS<L, P, G, F>
where
    F: ArgminFloat,
{
    /// Construct a new instance of [`LBFGS`]
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::quasinewton::LBFGS;
    /// # let linesearch = ();
    /// let lbfgs: LBFGS<_, Vec<f64>, Vec<f64>,  f64> = LBFGS::new(linesearch, 5);
    /// ```
    pub fn new(linesearch: L, m: usize) -> Self {
        LBFGS {
            linesearch,
            m,
            s: VecDeque::with_capacity(m),
            y: VecDeque::with_capacity(m),
            tol_grad: F::epsilon().sqrt(),
            tol_cost: F::epsilon(),
            l1_coeff: None,
            l1_prev_unreg_grad: None,
        }
    }

    /// The algorithm stops if the norm of the gradient is below `tol_grad`.
    ///
    /// The provided value must be non-negative. Defaults to `sqrt(EPSILON)`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::quasinewton::LBFGS;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// # let linesearch = ();
    /// let lbfgs: LBFGS<_, Vec<f64>, Vec<f64>,  f64> = LBFGS::new(linesearch, 3).with_tolerance_grad(1e-6)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_tolerance_grad(mut self, tol_grad: F) -> Result<Self, Error> {
        if tol_grad < float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`L-BFGS`: gradient tolerance must be >= 0."
            ));
        }
        self.tol_grad = tol_grad;
        Ok(self)
    }

    /// Sets tolerance for the stopping criterion based on the change of the cost stopping criterion
    ///
    /// The provided value must be non-negative. Defaults to `EPSILON`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::quasinewton::LBFGS;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// # let linesearch = ();
    /// let lbfgs: LBFGS<_, Vec<f64>, Vec<f64>, f64> = LBFGS::new(linesearch, 3).with_tolerance_cost(1e-6)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_tolerance_cost(mut self, tol_cost: F) -> Result<Self, Error> {
        if tol_cost < float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`L-BFGS`: cost tolerance must be >= 0."
            ));
        }
        self.tol_cost = tol_cost;
        Ok(self)
    }

    /// Activates L1-regularization with coefficient `l1_coeff`.
    ///
    /// Parameter `l1_coeff` must be `> 0.0`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::quasinewton::LBFGS;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// # let linesearch = ();
    /// let lbfgs: LBFGS<_, Vec<f64>, Vec<f64>, f64> = LBFGS::new(linesearch, 3).with_l1_regularization(1.0)?;
    /// # Ok(())
    /// # }
    pub fn with_l1_regularization(mut self, l1_coeff: F) -> Result<Self, Error> {
        if l1_coeff <= float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`L-BFGS`: coefficient of L1-regularization must be > 0."
            ));
        }
        self.l1_coeff = Some(l1_coeff);
        Ok(self)
    }
}

/// Wrapper problem for supporting constrained line search.
struct LineSearchProblem<O, P, G, F> {
    problem: O,
    xi: Option<P>,
    l1_coeff: Option<F>,
    phantom: PhantomData<G>,
}

impl<O, P, G, F> LineSearchProblem<O, P, G, F>
where
    P: ArgminSub<F, P>,
    F: ArgminFloat,
{
    fn new(problem: O) -> Self {
        Self {
            problem,
            xi: None,
            l1_coeff: None,
            phantom: PhantomData,
        }
    }

    fn with_l1_constraint(&mut self, l1_coeff: F, param: &P, pseudo_gradient: &G)
    where
        P: ArgminZeroLike
            + ArgminMinMax
            + ArgminSignum
            + ArgminAdd<P, P>
            + ArgminAdd<F, P>
            + ArgminMul<P, P>
            + ArgminSub<F, P>
            + ArgminMul<G, P>,
    {
        let zeros = param.zero_like();
        let sig_param = P::max(&param.sub(&F::min_positive_value()).signum(), &zeros).add(&P::min(
            &param.add(&F::min_positive_value()).signum(),
            &zeros,
        ));
        self.xi = Some(
            sig_param.add(
                &sig_param
                    .mul(&sig_param)
                    .sub(&float!(1.0))
                    .mul(pseudo_gradient),
            ),
        );
        self.l1_coeff = Some(l1_coeff);
    }
}

impl<O, P, G, F> CostFunction for LineSearchProblem<O, P, G, F>
where
    O: CostFunction<Param = P, Output = F>,
    P: ArgminMul<P, P> + ArgminMinMax + ArgminSignum + ArgminZeroLike + ArgminL1Norm<F>,
    F: ArgminFloat,
{
    type Param = P;
    type Output = F;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        if let Some(xi) = self.xi.as_ref() {
            let zeros = param.zero_like();
            let param = P::max(&param.mul(xi).signum(), &zeros).mul(param);
            let cost = self.problem.cost(&param)?;
            Ok(cost + self.l1_coeff.unwrap() * param.l1_norm())
        } else {
            self.problem.cost(param)
        }
    }
}

impl<O, P, G, F> Gradient for LineSearchProblem<O, P, G, F>
where
    O: Gradient<Param = P, Gradient = G>,
    P: ArgminAdd<F, P>
        + ArgminMul<P, P>
        + ArgminMul<F, P>
        + ArgminSub<F, P>
        + ArgminMinMax
        + ArgminSignum
        + ArgminZeroLike,
    G: ArgminAdd<P, G> + ArgminZeroLike + ArgminMinMax + ArgminAdd<G, G>,
    F: ArgminFloat,
{
    type Param = P;
    type Gradient = G;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, Error> {
        if let Some(xi) = self.xi.as_ref() {
            let zeros = param.zero_like();
            let param = P::max(&param.mul(xi).signum(), &zeros).mul(param);
            let gradient = self.problem.gradient(&param)?;
            Ok(calculate_pseudo_gradient(
                self.l1_coeff.unwrap(),
                &param,
                &gradient,
            ))
        } else {
            self.problem.gradient(param)
        }
    }
}

impl<O, L, P, G, F> Solver<O, IterState<P, G, (), (), F>> for LBFGS<L, P, G, F>
where
    O: CostFunction<Param = P, Output = F> + Gradient<Param = P, Gradient = G>,
    P: Clone
        + std::fmt::Debug
        + SerializeAlias
        + DeserializeOwnedAlias
        + ArgminSub<P, P>
        + ArgminSub<F, P>
        + ArgminAdd<P, P>
        + ArgminAdd<F, P>
        + ArgminDot<G, F>
        + ArgminMul<F, P>
        + ArgminMul<P, P>
        + ArgminMul<G, P>
        + ArgminL1Norm<F>
        + ArgminSignum
        + ArgminZeroLike
        + ArgminMinMax,
    G: Clone
        + std::fmt::Debug
        + SerializeAlias
        + DeserializeOwnedAlias
        + ArgminNorm<F>
        + ArgminSub<G, G>
        + ArgminAdd<G, G>
        + ArgminAdd<P, G>
        + ArgminDot<G, F>
        + ArgminDot<P, F>
        + ArgminMul<F, G>
        + ArgminMul<F, P>
        + ArgminZeroLike
        + ArgminMinMax,
    L: Clone + LineSearch<P, F> + Solver<LineSearchProblem<O, P, G, F>, IterState<P, G, (), (), F>>,
    F: ArgminFloat,
{
    const NAME: &'static str = "L-BFGS";

    fn init(
        &mut self,
        problem: &mut Problem<O>,
        mut state: IterState<P, G, (), (), F>,
    ) -> Result<(IterState<P, G, (), (), F>, Option<KV>), Error> {
        let param = state.take_param().ok_or_else(argmin_error_closure!(
            NotInitialized,
            concat!(
                "`L-BFGS` requires an initial parameter vector. ",
                "Please provide an initial guess via `Executor`s `configure` method."
            )
        ))?;

        let cost = state.get_cost();
        let cost = if cost.is_infinite() {
            if let Some(l1_coeff) = self.l1_coeff {
                problem.cost(&param)? + l1_coeff * param.l1_norm()
            } else {
                problem.cost(&param)?
            }
        } else {
            cost
        };

        let grad = state
            .take_gradient()
            .map(Result::Ok)
            .unwrap_or_else(|| problem.gradient(&param))?;

        Ok((state.param(param).cost(cost).gradient(grad), None))
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        mut state: IterState<P, G, (), (), F>,
    ) -> Result<(IterState<P, G, (), (), F>, Option<KV>), Error> {
        let param = state.take_param().ok_or_else(argmin_error_closure!(
            PotentialBug,
            "`L-BFGS`: Parameter vector in state not set."
        ))?;
        let cur_cost = state.get_cost();

        // If L1 regularization is enabled, the state contains pseudo gradient.
        let mut prev_grad = state.take_gradient().ok_or_else(argmin_error_closure!(
            PotentialBug,
            "`L-BFGS`: Gradient in state not set."
        ))?;
        if let Some(l1_coeff) = self.l1_coeff {
            if self.l1_prev_unreg_grad.is_none() {
                self.l1_prev_unreg_grad = Some(prev_grad.clone());
                prev_grad = calculate_pseudo_gradient(l1_coeff, &param, &prev_grad)
            }
        }

        let gamma: F = if let (Some(sk), Some(yk)) = (self.s.back(), self.y.back()) {
            sk.dot(yk) / yk.dot(yk)
        } else {
            float!(1.0)
        };

        // L-BFGS two-loop recursion
        let mut q = prev_grad.clone();
        let cur_m = self.s.len();
        let mut alpha: Vec<F> = vec![float!(0.0); cur_m];
        let mut rho: Vec<F> = vec![float!(0.0); cur_m];
        for (i, (sk, yk)) in self.s.iter().rev().zip(self.y.iter().rev()).enumerate() {
            let yksk: F = yk.dot(sk);
            let rho_t = float!(1.0) / yksk;
            let skq: F = sk.dot(&q);
            let alpha_t = skq.mul(rho_t);
            q = q.sub(&yk.mul(&alpha_t));
            rho[cur_m - i - 1] = rho_t;
            alpha[cur_m - i - 1] = alpha_t;
        }
        let mut r: P = q.mul(&gamma);
        for (i, (sk, yk)) in self.s.iter().zip(self.y.iter()).enumerate() {
            let beta: F = yk.dot(&r);
            let beta = beta.mul(rho[i]);
            r = r.add(&sk.mul(&(alpha[i] - beta)));
        }

        let mut line_problem = LineSearchProblem::new(problem.take_problem().unwrap());
        let d = if let Some(l1_coeff) = self.l1_coeff {
            line_problem.with_l1_constraint(l1_coeff, &param, &prev_grad);
            let zeros = r.zero_like();
            P::max(&r.mul(&prev_grad).signum(), &zeros)
                .mul(&r)
                .mul(&float!(-1.0))
        } else {
            r.mul(&float!(-1.0))
        };

        self.linesearch.search_direction(d);

        // Run solver
        let OptimizationResult {
            problem: mut line_problem,
            state: mut linesearch_state,
            ..
        } = Executor::new(line_problem, self.linesearch.clone())
            .configure(|config| {
                config
                    .param(param.clone())
                    .gradient(prev_grad.clone())
                    .cost(cur_cost)
            })
            .ctrlc(false)
            .run()?;

        let mut xk1 = linesearch_state.take_param().unwrap();
        let next_cost = linesearch_state.get_cost();

        // take back problem and take care of function evaluation counts
        let mut internal_line_problem = line_problem.take_problem().unwrap();
        let xi = internal_line_problem.xi.take();
        problem.problem = Some(internal_line_problem.problem);
        problem.consume_func_counts(line_problem);
        if let Some(xi) = xi {
            let zeros = xk1.zero_like();
            xk1 = P::max(&xk1.mul(&xi).signum(), &zeros).mul(&xk1);
        }

        if state.get_iter() >= self.m as u64 {
            self.s.pop_front();
            self.y.pop_front();
        }

        let grad = problem.gradient(&xk1)?;

        self.s.push_back(xk1.sub(&param));
        let grad = if let Some(l1_coeff) = self.l1_coeff {
            // Stores unregularized gradient and returns L1 gradient.
            let pseudo_grad = calculate_pseudo_gradient(l1_coeff, &param, &grad);
            self.y
                .push_back(grad.sub(self.l1_prev_unreg_grad.as_ref().unwrap()));
            self.l1_prev_unreg_grad = Some(grad);
            pseudo_grad
        } else {
            self.y.push_back(grad.sub(&prev_grad));
            grad
        };

        Ok((
            state.param(xk1).cost(next_cost).gradient(grad),
            Some(make_kv!("gamma" => gamma;)),
        ))
    }

    fn terminate(&mut self, state: &IterState<P, G, (), (), F>) -> TerminationReason {
        if state.get_gradient().unwrap().norm() < self.tol_grad {
            return TerminationReason::TargetPrecisionReached;
        }
        if (state.get_prev_cost() - state.get_cost()).abs() < self.tol_cost {
            return TerminationReason::NoChangeInCost;
        }
        TerminationReason::NotTerminated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{
        test_utils::{TestProblem, TestSparseProblem},
        ArgminError, IterState, State,
    };
    use crate::solver::linesearch::MoreThuenteLineSearch;
    use crate::test_trait_impl;

    test_trait_impl!(
        lbfgs,
        LBFGS<MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64>, Vec<f64>, Vec<f64>, f64>
    );

    #[test]
    fn test_new() {
        #[derive(Eq, PartialEq, Debug)]
        struct MyFakeLineSearch {}

        let lbfgs: LBFGS<_, Vec<f64>, Vec<f64>, f64> = LBFGS::new(MyFakeLineSearch {}, 3);
        let LBFGS {
            linesearch,
            tol_grad,
            tol_cost,
            m,
            s,
            y,
            l1_coeff,
            l1_prev_unreg_grad,
        } = lbfgs;

        assert_eq!(linesearch, MyFakeLineSearch {});
        assert_eq!(tol_grad.to_ne_bytes(), f64::EPSILON.sqrt().to_ne_bytes());
        assert_eq!(tol_cost.to_ne_bytes(), f64::EPSILON.to_ne_bytes());
        assert_eq!(m, 3);
        assert!(s.capacity() >= 3);
        assert!(y.capacity() >= 3);
        assert!(l1_coeff.is_none());
        assert!(l1_prev_unreg_grad.is_none());
    }

    #[test]
    fn test_with_tolerance_grad() {
        #[derive(Eq, PartialEq, Debug, Clone, Copy)]
        struct MyFakeLineSearch {}

        // correct parameters
        for tol in [1e-6, 0.0, 1e-2, 1.0, 2.0] {
            let lbfgs: LBFGS<_, Vec<f64>, Vec<f64>, f64> = LBFGS::new(MyFakeLineSearch {}, 3);
            let res = lbfgs.with_tolerance_grad(tol);
            assert!(res.is_ok());

            let nm = res.unwrap();
            assert_eq!(nm.tol_grad.to_ne_bytes(), tol.to_ne_bytes());
        }

        // incorrect parameters
        for tol in [-f64::EPSILON, -1.0, -100.0, -42.0] {
            let lbfgs: LBFGS<_, Vec<f64>, Vec<f64>, f64> = LBFGS::new(MyFakeLineSearch {}, 3);
            let res = lbfgs.with_tolerance_grad(tol);
            assert_error!(
                res,
                ArgminError,
                "Invalid parameter: \"`L-BFGS`: gradient tolerance must be >= 0.\""
            );
        }
    }

    #[test]
    fn test_with_tolerance_cost() {
        #[derive(Eq, PartialEq, Debug, Clone, Copy)]
        struct MyFakeLineSearch {}

        // correct parameters
        for tol in [1e-6, 0.0, 1e-2, 1.0, 2.0] {
            let lbfgs: LBFGS<_, Vec<f64>, Vec<f64>, f64> = LBFGS::new(MyFakeLineSearch {}, 3);
            let res = lbfgs.with_tolerance_cost(tol);
            assert!(res.is_ok());

            let nm = res.unwrap();
            assert_eq!(nm.tol_cost.to_ne_bytes(), tol.to_ne_bytes());
        }

        // incorrect parameters
        for tol in [-f64::EPSILON, -1.0, -100.0, -42.0] {
            let lbfgs: LBFGS<_, Vec<f64>, Vec<f64>, f64> = LBFGS::new(MyFakeLineSearch {}, 3);
            let res = lbfgs.with_tolerance_cost(tol);
            assert_error!(
                res,
                ArgminError,
                "Invalid parameter: \"`L-BFGS`: cost tolerance must be >= 0.\""
            );
        }
    }

    #[test]
    fn test_init() {
        let linesearch = MoreThuenteLineSearch::new().with_c(1e-4, 0.9).unwrap();

        let param: Vec<f64> = vec![-1.0, 1.0];

        let mut lbfgs: LBFGS<_, Vec<f64>, Vec<f64>, f64> = LBFGS::new(linesearch, 3);

        // Forgot to initialize the parameter vector
        let state: IterState<Vec<f64>, Vec<f64>, (), (), f64> = IterState::new();
        let problem = TestProblem::new();
        let res = lbfgs.init(&mut Problem::new(problem), state);
        assert_error!(
            res,
            ArgminError,
            concat!(
                "Not initialized: \"`L-BFGS` requires an initial parameter vector. Please ",
                "provide an initial guess via `Executor`s `configure` method.\""
            )
        );

        // All good.
        let state: IterState<Vec<f64>, Vec<f64>, (), (), f64> =
            IterState::new().param(param.clone());
        let problem = TestProblem::new();
        let (mut state_out, kv) = lbfgs.init(&mut Problem::new(problem), state).unwrap();

        assert!(kv.is_none());

        let s_param = state_out.take_param().unwrap();

        for (s, p) in s_param.iter().zip(param.iter()) {
            assert_eq!(s.to_ne_bytes(), p.to_ne_bytes());
        }

        let s_grad = state_out.take_gradient().unwrap();

        for (s, p) in s_grad.iter().zip(param.iter()) {
            assert_eq!(s.to_ne_bytes(), p.to_ne_bytes());
        }

        assert_eq!(state_out.get_cost().to_ne_bytes(), 1.0f64.to_ne_bytes())
    }

    #[test]
    fn test_init_provided_cost() {
        let linesearch = MoreThuenteLineSearch::new().with_c(1e-4, 0.9).unwrap();

        let param: Vec<f64> = vec![-1.0, 1.0];

        let mut lbfgs: LBFGS<_, Vec<f64>, Vec<f64>, f64> = LBFGS::new(linesearch, 3);

        let state: IterState<Vec<f64>, Vec<f64>, (), (), f64> =
            IterState::new().param(param).cost(1234.0);

        let problem = TestProblem::new();
        let (state_out, kv) = lbfgs.init(&mut Problem::new(problem), state).unwrap();

        assert!(kv.is_none());

        assert_eq!(state_out.get_cost().to_ne_bytes(), 1234.0f64.to_ne_bytes())
    }

    #[test]
    fn test_init_provided_grad() {
        let linesearch = MoreThuenteLineSearch::new().with_c(1e-4, 0.9).unwrap();

        let param: Vec<f64> = vec![-1.0, 1.0];
        let gradient: Vec<f64> = vec![4.0, 9.0];

        let mut lbfgs: LBFGS<_, Vec<f64>, Vec<f64>, f64> = LBFGS::new(linesearch, 3);

        let state: IterState<Vec<f64>, Vec<f64>, (), (), f64> =
            IterState::new().param(param).gradient(gradient.clone());

        let problem = TestProblem::new();
        let (mut state_out, kv) = lbfgs.init(&mut Problem::new(problem), state).unwrap();

        assert!(kv.is_none());

        let s_grad = state_out.take_gradient().unwrap();

        for (s, g) in s_grad.iter().zip(gradient.iter()) {
            assert_eq!(s.to_ne_bytes(), g.to_ne_bytes());
        }
    }

    #[test]
    fn test_l1_regularization() {
        {
            let linesearch = MoreThuenteLineSearch::new().with_c(1e-4, 0.9).unwrap();

            let param: Vec<f64> = vec![0.0; 4];

            let lbfgs: LBFGS<_, Vec<f64>, Vec<f64>, f64> = LBFGS::new(linesearch, 3);

            let cost = TestSparseProblem::new();
            let res = Executor::new(cost, lbfgs)
                .configure(|state| state.param(param).max_iters(2))
                .run()
                .unwrap();

            let result_param = res.state.param.unwrap();

            assert!((result_param[0] - 0.5).abs() > 1e-6);
            assert!((result_param[1]).abs() > 1e-6);
            assert!((result_param[2] + 0.5).abs() > 1e-6);
            assert!((result_param[3]).abs() > 1e-6);
        }
        {
            let linesearch = MoreThuenteLineSearch::new().with_c(1e-4, 0.9).unwrap();

            let param: Vec<f64> = vec![0.0; 4];

            let lbfgs: LBFGS<_, Vec<f64>, Vec<f64>, f64> = LBFGS::new(linesearch, 3)
                .with_l1_regularization(2.0)
                .unwrap();

            let cost = TestSparseProblem::new();
            let res = Executor::new(cost, lbfgs)
                .configure(|state| state.param(param).max_iters(2))
                .run()
                .unwrap();

            let result_param = res.state.param.unwrap();
            dbg!(&result_param);

            assert!((result_param[0] - 0.5).abs() < 1e-6);
            assert!((result_param[1]).abs() < 1e-6);
            assert!((result_param[2] + 0.5).abs() < 1e-6);
            assert!((result_param[3]).abs() < 1e-6);
        }
    }
}
