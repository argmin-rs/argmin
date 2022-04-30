// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::{
    ArgminFloat, DeserializeOwnedAlias, Error, Executor, Gradient, Hessian, IterState, LineSearch,
    Operator, OptimizationResult, Problem, SerializeAlias, Solver, State, TerminationReason, KV,
};
use crate::solver::conjugategradient::ConjugateGradient;
use argmin_math::{
    ArgminConj, ArgminDot, ArgminMul, ArgminNorm, ArgminScaledAdd, ArgminSub, ArgminZeroLike,
};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// # Newton-Conjugate-Gradient (Newton-CG) method
///
/// The Newton-CG method (also called truncated Newton method) uses a modified CG to solve the
/// Newton equations approximately. After a search direction is found, a line search is performed.
///
/// TODO: Stop when search direction is close to 0
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
    /// Constructor
    pub fn new(linesearch: L) -> Self {
        NewtonCG {
            linesearch,
            curvature_threshold: float!(0.0),
            tol: F::epsilon(),
        }
    }

    /// Set curvature threshold
    #[must_use]
    pub fn curvature_threshold(mut self, threshold: F) -> Self {
        self.curvature_threshold = threshold;
        self
    }

    /// Set tolerance for the stopping criterion based on cost difference
    pub fn with_tolerance(mut self, tol: F) -> Result<Self, Error> {
        if tol <= float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "Newton-CG: tol must be positive."
            ));
        }
        self.tol = tol;
        Ok(self)
    }
}

impl<O, L, P, G, H, F> Solver<O, IterState<P, G, (), H, F>> for NewtonCG<L, F>
where
    O: Gradient<Param = P, Gradient = G> + Hessian<Param = P, Hessian = H>,
    P: Clone
        + SerializeAlias
        + DeserializeOwnedAlias
        + ArgminSub<P, P>
        + ArgminDot<P, F>
        + ArgminScaledAdd<P, F, P>
        + ArgminMul<F, P>
        + ArgminConj
        + ArgminZeroLike,
    G: SerializeAlias + DeserializeOwnedAlias + ArgminNorm<F> + ArgminMul<F, P>,
    H: Clone + SerializeAlias + DeserializeOwnedAlias + ArgminDot<P, P>,
    L: Clone + LineSearch<P, F> + Solver<O, IterState<P, G, (), (), F>>,
    F: ArgminFloat + ArgminNorm<F>,
{
    const NAME: &'static str = "Newton-CG";

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        mut state: IterState<P, G, (), H, F>,
    ) -> Result<(IterState<P, G, (), H, F>, Option<KV>), Error> {
        let param = state.take_param().unwrap();
        let grad = problem.gradient(&param)?;
        let hessian = problem.hessian(&param)?;

        // Solve CG subproblem
        let cg_problem: CGSubProblem<P, H> = CGSubProblem::new(hessian.clone());
        let mut cg_problem = Problem::new(cg_problem);

        let mut x_p = param.zero_like();
        let mut x: P = param.zero_like();
        let mut cg = ConjugateGradient::new(grad.mul(&(float!(-1.0))));

        let cg_state = IterState::new().param(x_p.clone());
        let (mut cg_state, _) = cg.init(&mut cg_problem, cg_state)?;
        let grad_norm = grad.norm();
        for iter in 0.. {
            let (state_tmp, _) = cg.next_iter(&mut cg_problem, cg_state)?;
            cg_state = state_tmp;
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
            if cost <= float!(0.5).min(grad_norm.sqrt()) * grad_norm {
                break;
            }
            cg_state = cg_state.param(x.clone()).cost(cost);
            x_p = x.clone();
        }

        // perform line search
        self.linesearch.search_direction(x);

        let line_cost = state.get_cost();

        // Run solver
        let OptimizationResult {
            problem: line_problem,
            state: mut linesearch_state,
            ..
        } = Executor::new(problem.take_problem().unwrap(), self.linesearch.clone())
            .configure(|config| config.param(param).grad(grad).cost(line_cost))
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

    fn terminate(&mut self, state: &IterState<P, G, (), H, F>) -> TerminationReason {
        if (state.get_cost() - state.get_prev_cost()).abs() < self.tol {
            TerminationReason::NoChangeInCost
        } else {
            TerminationReason::NotTerminated
        }
    }
}

#[derive(Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
struct CGSubProblem<P, H> {
    hessian: H,
    phantom: std::marker::PhantomData<P>,
}

impl<T, H> CGSubProblem<T, H> {
    /// constructor
    pub fn new(hessian: H) -> Self {
        CGSubProblem {
            hessian,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<P, H> Operator for CGSubProblem<P, H>
where
    H: ArgminDot<P, P>,
    P: Clone + SerializeAlias + DeserializeOwnedAlias,
{
    type Param = P;
    type Output = P;

    fn apply(&self, p: &P) -> Result<P, Error> {
        Ok(self.hessian.dot(p))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::linesearch::MoreThuenteLineSearch;
    use crate::test_trait_impl;

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
}
