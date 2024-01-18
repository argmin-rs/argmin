// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::{
    ArgminFloat, CostFunction, Error, Executor, Gradient, IterState, LineSearch,
    OptimizationResult, Problem, Solver, TerminationReason, TerminationStatus, KV,
};
use argmin_math::{ArgminAdd, ArgminDot, ArgminL2Norm, ArgminMul, ArgminSub};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// # Davidon-Fletcher-Powell (DFP) method
///
/// The Davidon-Fletcher-Powell algorithm (DFP) is a method for solving unconstrained nonlinear
/// optimization problems.
///
/// The algorithm requires a line search which is provided via the constructor. Additionally an
/// initial guess for the parameter vector and an initial inverse Hessian is required, which are to
/// be provided via the [`configure`](`crate::core::Executor::configure`) method of the
/// [`Executor`](`crate::core::Executor`) (See [`IterState`], in particular [`IterState::param`]
/// and [`IterState::inv_hessian`]).
/// In the same way the initial gradient and cost function corresponding to the initial parameter
/// vector can be provided. If these are not provided, they will be computed during initialization
/// of the algorithm.
///
/// A tolerance on the gradient can be configured with
/// [`with_tolerance_grad`](`DFP::with_tolerance_grad`): If the norm of the gradient is below
/// said tolerance, the algorithm stops. It defaults to `sqrt(EPSILON)`.
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
pub struct DFP<L, F> {
    /// line search
    linesearch: L,
    /// Tolerance for the stopping criterion based on the change of the norm on the gradient
    tol_grad: F,
}

impl<L, F> DFP<L, F>
where
    F: ArgminFloat,
{
    /// Construct a new instance of [`DFP`]
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::quasinewton::DFP;
    /// # let linesearch = ();
    /// let dfp: DFP<_, f64> = DFP::new(linesearch);
    /// ```
    pub fn new(linesearch: L) -> Self {
        DFP {
            linesearch,
            tol_grad: F::epsilon().sqrt(),
        }
    }

    /// The algorithm stops if the norm of the gradient is below `tol_grad`.
    ///
    /// The provided value must be non-negative. Defaults to `sqrt(EPSILON)`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::quasinewton::DFP;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// # let linesearch = ();
    /// let dfp: DFP<_, f64> = DFP::new(linesearch).with_tolerance_grad(1e-6)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_tolerance_grad(mut self, tol_grad: F) -> Result<Self, Error> {
        if tol_grad < float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`DFP`: gradient tolerance must be >= 0."
            ));
        }
        self.tol_grad = tol_grad;
        Ok(self)
    }
}

impl<O, L, P, G, H, F> Solver<O, IterState<P, G, (), H, (), F>> for DFP<L, F>
where
    O: CostFunction<Param = P, Output = F> + Gradient<Param = P, Gradient = G>,
    P: Clone + ArgminSub<P, P> + ArgminDot<G, F> + ArgminDot<P, H> + ArgminMul<F, P>,
    G: Clone + ArgminSub<G, G> + ArgminL2Norm<F> + ArgminDot<P, F>,
    H: Clone + ArgminSub<H, H> + ArgminDot<G, P> + ArgminAdd<H, H> + ArgminMul<F, H>,
    L: Clone + LineSearch<P, F> + Solver<O, IterState<P, G, (), (), (), F>>,
    F: ArgminFloat,
{
    const NAME: &'static str = "DFP";

    fn init(
        &mut self,
        problem: &mut Problem<O>,
        mut state: IterState<P, G, (), H, (), F>,
    ) -> Result<(IterState<P, G, (), H, (), F>, Option<KV>), Error> {
        let param = state.take_param().ok_or_else(argmin_error_closure!(
            NotInitialized,
            concat!(
                "`DFP` requires an initial parameter vector. ",
                "Please provide an initial guess via `Executor`s `configure` method."
            )
        ))?;

        let inv_hessian = state.take_inv_hessian().ok_or_else(argmin_error_closure!(
            NotInitialized,
            concat!(
                "`DFP` requires an initial inverse Hessian. ",
                "Please provide an initial guess via `Executor`s `configure` method."
            )
        ))?;

        let cost = state.get_cost();
        let cost = if cost.is_infinite() {
            problem.cost(&param)?
        } else {
            cost
        };

        let grad = state
            .take_gradient()
            .map(Result::Ok)
            .unwrap_or_else(|| problem.gradient(&param))?;

        Ok((
            state
                .param(param)
                .cost(cost)
                .gradient(grad)
                .inv_hessian(inv_hessian),
            None,
        ))
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        mut state: IterState<P, G, (), H, (), F>,
    ) -> Result<(IterState<P, G, (), H, (), F>, Option<KV>), Error> {
        let param = state.take_param().ok_or_else(argmin_error_closure!(
            PotentialBug,
            "`DFP`: Parameter vector in state not set."
        ))?;
        let cost = state.get_cost();

        let prev_grad = state.take_gradient().ok_or_else(argmin_error_closure!(
            PotentialBug,
            "`DFP`: Gradient in state not set."
        ))?;

        let inv_hessian = state.take_inv_hessian().ok_or_else(argmin_error_closure!(
            PotentialBug,
            "`DFP`: Inverse Hessian in state not set."
        ))?;

        let p = inv_hessian.dot(&prev_grad).mul(&float!(-1.0));

        self.linesearch.search_direction(p);

        let OptimizationResult {
            problem: line_problem,
            state: mut linesearch_state,
            ..
        } = Executor::new(problem.take_problem().unwrap(), self.linesearch.clone())
            .configure(|config| {
                config
                    .param(param.clone())
                    .gradient(prev_grad.clone())
                    .cost(cost)
            })
            .ctrlc(false)
            .run()?;

        let xk1 = linesearch_state
            .take_param()
            .ok_or_else(argmin_error_closure!(
                PotentialBug,
                "`DFP`: No parameters returned by line search."
            ))?;

        let next_cost = linesearch_state.get_cost();

        // take care of function eval counts
        problem.consume_problem(line_problem);

        let grad = problem.gradient(&xk1)?;
        let yk = grad.sub(&prev_grad);

        let sk = xk1.sub(&param);

        let yksk: F = yk.dot(&sk);

        let sksk: H = sk.dot(&sk);

        let tmp3: P = inv_hessian.dot(&yk);
        let tmp4: F = tmp3.dot(&yk);
        let tmp3: H = tmp3.dot(&tmp3);
        let tmp3: H = tmp3.mul(&(float!(1.0) / tmp4));

        let inv_hessian = inv_hessian.sub(&tmp3).add(&sksk.mul(&(float!(1.0) / yksk)));

        Ok((
            state
                .param(xk1)
                .cost(next_cost)
                .gradient(grad)
                .inv_hessian(inv_hessian),
            None,
        ))
    }

    fn terminate(&mut self, state: &IterState<P, G, (), H, (), F>) -> TerminationStatus {
        if state.get_gradient().unwrap().l2_norm() < self.tol_grad {
            return TerminationStatus::Terminated(TerminationReason::SolverConverged);
        }
        TerminationStatus::NotTerminated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{test_utils::TestProblem, ArgminError, IterState, State};
    use crate::solver::linesearch::MoreThuenteLineSearch;
    use crate::test_trait_impl;

    test_trait_impl!(
        dfp,
        DFP<MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64>, f64>
    );

    #[test]
    fn test_new() {
        #[derive(Eq, PartialEq, Debug)]
        struct MyFakeLineSearch {}

        let dfp: DFP<_, f64> = DFP::new(MyFakeLineSearch {});
        let DFP {
            linesearch,
            tol_grad,
        } = dfp;

        assert_eq!(linesearch, MyFakeLineSearch {});
        assert_eq!(tol_grad.to_ne_bytes(), f64::EPSILON.sqrt().to_ne_bytes());
    }

    #[test]
    fn test_with_tolerance_grad() {
        #[derive(Eq, PartialEq, Debug, Clone, Copy)]
        struct MyFakeLineSearch {}

        // correct parameters
        for tol in [1e-6, 0.0, 1e-2, 1.0, 2.0] {
            let dfp: DFP<_, f64> = DFP::new(MyFakeLineSearch {});
            let res = dfp.with_tolerance_grad(tol);
            assert!(res.is_ok());

            let nm = res.unwrap();
            assert_eq!(nm.tol_grad.to_ne_bytes(), tol.to_ne_bytes());
        }

        // incorrect parameters
        for tol in [-f64::EPSILON, -1.0, -100.0, -42.0] {
            let dfp: DFP<_, f64> = DFP::new(MyFakeLineSearch {});
            let res = dfp.with_tolerance_grad(tol);
            assert_error!(
                res,
                ArgminError,
                "Invalid parameter: \"`DFP`: gradient tolerance must be >= 0.\""
            );
        }
    }

    #[test]
    fn test_init() {
        let linesearch = MoreThuenteLineSearch::new().with_c(1e-4, 0.9).unwrap();

        let param: Vec<f64> = vec![-1.0, 1.0];
        let inv_hessian: Vec<Vec<f64>> = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

        let mut dfp: DFP<_, f64> = DFP::new(linesearch);

        // Forgot to initialize the parameter vector
        let state: IterState<Vec<f64>, Vec<f64>, (), Vec<Vec<f64>>, (), f64> = IterState::new();
        let problem = TestProblem::new();
        let res = dfp.init(&mut Problem::new(problem), state);
        assert_error!(
            res,
            ArgminError,
            concat!(
                "Not initialized: \"`DFP` requires an initial parameter vector. Please ",
                "provide an initial guess via `Executor`s `configure` method.\""
            )
        );

        // Forgot initial inverse Hessian guess
        let state: IterState<Vec<f64>, Vec<f64>, (), Vec<Vec<f64>>, (), f64> =
            IterState::new().param(param.clone());
        let problem = TestProblem::new();
        let res = dfp.init(&mut Problem::new(problem), state);

        assert_error!(
            res,
            ArgminError,
            concat!(
                "Not initialized: \"`DFP` requires an initial inverse Hessian. Please ",
                "provide an initial guess via `Executor`s `configure` method.\""
            )
        );

        // All good.
        let state: IterState<Vec<f64>, Vec<f64>, (), Vec<Vec<f64>>, (), f64> = IterState::new()
            .param(param.clone())
            .inv_hessian(inv_hessian.clone());
        let problem = TestProblem::new();
        let (mut state_out, kv) = dfp.init(&mut Problem::new(problem), state).unwrap();

        assert!(kv.is_none());

        let s_param = state_out.take_param().unwrap();

        for (s, p) in s_param.iter().zip(param.iter()) {
            assert_eq!(s.to_ne_bytes(), p.to_ne_bytes());
        }

        let s_grad = state_out.take_gradient().unwrap();

        for (s, p) in s_grad.iter().zip(param.iter()) {
            assert_eq!(s.to_ne_bytes(), p.to_ne_bytes());
        }

        let s_inv_hessian = state_out.take_inv_hessian().unwrap();

        for (s, h) in s_inv_hessian
            .iter()
            .flatten()
            .zip(inv_hessian.iter().flatten())
        {
            assert_eq!(s.to_ne_bytes(), h.to_ne_bytes());
        }

        assert_eq!(state_out.get_cost().to_ne_bytes(), 1.0f64.to_ne_bytes())
    }

    #[test]
    fn test_init_provided_cost() {
        let linesearch = MoreThuenteLineSearch::new().with_c(1e-4, 0.9).unwrap();

        let param: Vec<f64> = vec![-1.0, 1.0];
        let inv_hessian: Vec<Vec<f64>> = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

        let mut dfp: DFP<_, f64> = DFP::new(linesearch);

        let state: IterState<Vec<f64>, Vec<f64>, (), Vec<Vec<f64>>, (), f64> = IterState::new()
            .param(param)
            .inv_hessian(inv_hessian)
            .cost(1234.0);

        let problem = TestProblem::new();
        let (state_out, kv) = dfp.init(&mut Problem::new(problem), state).unwrap();

        assert!(kv.is_none());

        assert_eq!(state_out.get_cost().to_ne_bytes(), 1234.0f64.to_ne_bytes())
    }

    #[test]
    fn test_init_provided_grad() {
        let linesearch = MoreThuenteLineSearch::new().with_c(1e-4, 0.9).unwrap();

        let param: Vec<f64> = vec![-1.0, 1.0];
        let gradient: Vec<f64> = vec![4.0, 9.0];
        let inv_hessian: Vec<Vec<f64>> = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

        let mut dfp: DFP<_, f64> = DFP::new(linesearch);

        let state: IterState<Vec<f64>, Vec<f64>, (), Vec<Vec<f64>>, (), f64> = IterState::new()
            .param(param)
            .inv_hessian(inv_hessian)
            .gradient(gradient.clone());

        let problem = TestProblem::new();
        let (mut state_out, kv) = dfp.init(&mut Problem::new(problem), state).unwrap();

        assert!(kv.is_none());

        let s_grad = state_out.take_gradient().unwrap();

        for (s, g) in s_grad.iter().zip(gradient.iter()) {
            assert_eq!(s.to_ne_bytes(), g.to_ne_bytes());
        }
    }
}
