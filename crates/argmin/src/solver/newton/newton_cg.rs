// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::{
    ArgminFloat, Error, Executor, Gradient, Hessian, IterState, LineSearch, Operator,
    OptimizationResult, Problem, Solver, State, TerminationReason, TerminationStatus, KV,
};
use crate::solver::conjugategradient::ConjugateGradient;
use argmin_math::{
    ArgminConj, ArgminDot, ArgminL2Norm, ArgminMul, ArgminScaledAdd, ArgminSub, ArgminZeroLike,
};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// # Newton-Conjugate-Gradient (Newton-CG) method
///
/// The Newton-CG method (also called truncated Newton method) uses a modified CG to approximately
/// solve the Newton equations. After a search direction is found, a line search is performed.
///
/// ## Requirements on the optimization problem
///
/// The optimization problem is required to implement [`Gradient`] and [`Hessian`].
///
/// ## Reference
///
/// Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct NewtonCG<L, F> {
    /// line search
    linesearch: L,
    /// curvature_threshold
    curvature_threshold: F,
    /// Tolerance for the stopping criterion based on cost difference
    tol: F,
}

impl<L, F> NewtonCG<L, F>
where
    F: ArgminFloat,
{
    /// Construct a new instance of [`NewtonCG`]
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::newton::NewtonCG;
    /// # let linesearch = ();
    /// let ncg: NewtonCG<_, f64> = NewtonCG::new(linesearch);
    /// ```
    pub fn new(linesearch: L) -> Self {
        NewtonCG {
            linesearch,
            curvature_threshold: float!(0.0),
            tol: F::epsilon(),
        }
    }

    /// Set curvature threshold
    ///
    /// Defaults to 0.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::newton::NewtonCG;
    /// # let linesearch = ();
    /// let ncg: NewtonCG<_, f64> = NewtonCG::new(linesearch).with_curvature_threshold(1e-6);
    /// ```
    #[must_use]
    pub fn with_curvature_threshold(mut self, threshold: F) -> Self {
        self.curvature_threshold = threshold;
        self
    }

    /// Set tolerance for the stopping criterion based on cost difference
    ///
    /// Must be larger than 0 and defaults to EPSILON.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::newton::NewtonCG;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// # let linesearch = ();
    /// let ncg: NewtonCG<_, f64> = NewtonCG::new(linesearch).with_tolerance(1e-6)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_tolerance(mut self, tol: F) -> Result<Self, Error> {
        if tol <= float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`NewtonCG`: tol must be > 0."
            ));
        }
        self.tol = tol;
        Ok(self)
    }
}

impl<O, L, P, G, H, F> Solver<O, IterState<P, G, (), H, (), F>> for NewtonCG<L, F>
where
    O: Gradient<Param = P, Gradient = G> + Hessian<Param = P, Hessian = H>,
    P: Clone
        + ArgminSub<P, P>
        + ArgminDot<P, F>
        + ArgminScaledAdd<P, F, P>
        + ArgminMul<F, P>
        + ArgminConj
        + ArgminZeroLike,
    G: ArgminL2Norm<F> + ArgminMul<F, P>,
    H: Clone + ArgminDot<P, P>,
    L: Clone + LineSearch<P, F> + Solver<O, IterState<P, G, (), (), (), F>>,
    F: ArgminFloat + ArgminL2Norm<F>,
{
    const NAME: &'static str = "Newton-CG";

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        mut state: IterState<P, G, (), H, (), F>,
    ) -> Result<(IterState<P, G, (), H, (), F>, Option<KV>), Error> {
        let param = state.take_param().ok_or_else(argmin_error_closure!(
            NotInitialized,
            concat!(
                "`NewtonCG` requires an initial parameter vector. ",
                "Please provide an initial guess via `Executor`s `configure` method."
            )
        ))?;
        let grad = state
            .take_gradient()
            .map(Result::Ok)
            .unwrap_or_else(|| problem.gradient(&param))?;
        let hessian = state
            .take_hessian()
            .map(Result::Ok)
            .unwrap_or_else(|| problem.hessian(&param))?;

        // Solve CG subproblem
        let mut cg_problem = Problem::new(CGSubProblem::new(&hessian));

        let mut x_p = param.zero_like();
        let mut x = param.zero_like();
        let mut cg = ConjugateGradient::new(grad.mul(&(float!(-1.0))));

        let (mut cg_state, _): (IterState<_, _, _, _, _, _>, _) =
            cg.init(&mut cg_problem, IterState::new().param(x_p.clone()))?;

        let grad_norm_factor = float!(0.5).min(grad.l2_norm().sqrt()) * grad.l2_norm();

        for iter in 0.. {
            (cg_state, _) = cg.next_iter(&mut cg_problem, cg_state)?;

            let cost = cg_state.get_cost();

            x = cg_state.take_param().unwrap();
            let p = cg.get_prev_p()?;

            let curvature = p.dot(&hessian.dot(p));
            if curvature <= self.curvature_threshold {
                if iter == 0 {
                    x = grad.mul(&(float!(-1.0)));
                } else {
                    x = x_p;
                }
                break;
            }

            if cost <= grad_norm_factor {
                break;
            }

            cg_state = cg_state.param(x.clone()).cost(cost);
            x_p = x.clone();
        }

        // perform line search
        // TODO: Should the algorithm stop when search direction is close to 0?
        self.linesearch.search_direction(x);

        let line_cost = state.get_cost();

        // Run solver
        let OptimizationResult {
            problem: line_problem,
            state: mut linesearch_state,
            ..
        } = Executor::new(problem.take_problem().unwrap(), self.linesearch.clone())
            .configure(|state| state.param(param).gradient(grad).cost(line_cost))
            .ctrlc(false)
            .run()?;

        problem.consume_problem(line_problem);

        Ok((
            state
                .param(linesearch_state.take_param().unwrap())
                .cost(linesearch_state.get_cost()),
            None,
        ))
    }

    fn terminate(&mut self, state: &IterState<P, G, (), H, (), F>) -> TerminationStatus {
        if (state.get_cost() - state.get_prev_cost()).abs() < self.tol {
            TerminationStatus::Terminated(TerminationReason::SolverConverged)
        } else {
            TerminationStatus::NotTerminated
        }
    }
}

#[derive(Clone)]
struct CGSubProblem<'a, P, H> {
    hessian: &'a H,
    phantom: std::marker::PhantomData<P>,
}

impl<'a, P, H> CGSubProblem<'a, P, H> {
    /// Constructor
    fn new(hessian: &'a H) -> Self {
        CGSubProblem {
            hessian,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<'a, P, H> Operator for CGSubProblem<'a, P, H>
where
    H: ArgminDot<P, P>,
{
    type Param = P;
    type Output = P;

    fn apply(&self, p: &P) -> Result<P, Error> {
        Ok(self.hessian.dot(p))
    }
}

#[cfg(test)]
#[allow(clippy::let_unit_value)]
mod tests {
    use super::*;
    use crate::core::{test_utils::TestProblem, ArgminError};
    use crate::solver::linesearch::MoreThuenteLineSearch;

    test_trait_impl!(
        newton_cg,
        NewtonCG<MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64>, f64>
    );

    test_trait_impl!(cg_subproblem, CGSubProblem<Vec<f64>, Vec<Vec<f64>>>);

    #[test]
    fn test_tolerance() {
        let tol1: f64 = 1e-4;

        let linesearch: MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64> =
            MoreThuenteLineSearch::new();

        let NewtonCG { tol: t, .. }: NewtonCG<MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64>, f64> =
            NewtonCG::new(linesearch).with_tolerance(tol1).unwrap();

        assert!((t - tol1).abs() < std::f64::EPSILON);
    }

    #[test]
    fn test_new() {
        #[derive(Eq, PartialEq, Debug, Copy, Clone)]
        struct LineSearch {}
        let ls = LineSearch {};
        let ncg: NewtonCG<_, f64> = NewtonCG::new(ls);
        let NewtonCG {
            linesearch,
            curvature_threshold,
            tol,
        } = ncg;
        assert_eq!(linesearch, ls);
        assert_eq!(curvature_threshold.to_ne_bytes(), 0.0f64.to_ne_bytes());
        assert_eq!(tol.to_ne_bytes(), f64::EPSILON.to_ne_bytes());
    }

    #[test]
    fn test_with_curvature_threshold() {
        #[derive(Eq, PartialEq, Debug, Copy, Clone)]
        struct LineSearch {}
        let ls = LineSearch {};
        let ncg: NewtonCG<_, f64> = NewtonCG::new(ls).with_curvature_threshold(1e-6);
        let NewtonCG {
            linesearch,
            curvature_threshold,
            tol,
        } = ncg;
        assert_eq!(linesearch, ls);
        assert_eq!(curvature_threshold.to_ne_bytes(), 1e-6f64.to_ne_bytes());
        assert_eq!(tol.to_ne_bytes(), f64::EPSILON.to_ne_bytes());
    }

    #[test]
    fn test_with_tolerance() {
        let ls = ();
        for tolerance in [f64::EPSILON, 1.0, 10.0, 100.0] {
            let ncg: NewtonCG<_, f64> = NewtonCG::new(ls).with_tolerance(tolerance).unwrap();
            assert_eq!(ncg.tol.to_ne_bytes(), tolerance.to_ne_bytes());
        }

        for tolerance in [-f64::EPSILON, 0.0, -1.0] {
            let res = NewtonCG::new(ls).with_tolerance(tolerance);
            assert_error!(
                res,
                ArgminError,
                "Invalid parameter: \"`NewtonCG`: tol must be > 0.\""
            );
        }
    }

    #[test]
    fn test_next_iter_param_not_initialized() {
        use crate::solver::linesearch::{condition::ArmijoCondition, BacktrackingLineSearch};
        let ls = BacktrackingLineSearch::new(ArmijoCondition::new(0.9f64).unwrap());
        let mut ncg: NewtonCG<_, f64> = NewtonCG::new(ls);
        let res = ncg.next_iter(&mut Problem::new(TestProblem::new()), IterState::new());
        assert_error!(
            res,
            ArgminError,
            concat!(
                "Not initialized: \"`NewtonCG` requires an initial parameter vector. ",
                "Please provide an initial guess via `Executor`s `configure` method.\""
            )
        );
    }

    // TODO: Test next_iter.
}
