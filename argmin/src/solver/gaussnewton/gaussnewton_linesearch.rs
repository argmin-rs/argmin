// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # References:
//!
//! \[0\] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
//! Springer. ISBN 0-387-30303-0.

use crate::core::{
    ArgminError, ArgminFloat, CostFunction, DeserializeOwnedAlias, Error, Executor, Gradient,
    Hessian, IterState, Jacobian, LineSearch, Operator, OptimizationResult, Problem,
    SerializeAlias, Solver, TerminationReason, KV,
};
use argmin_math::{ArgminDot, ArgminInv, ArgminMul, ArgminNorm, ArgminTranspose};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// Gauss-Newton method with linesearch
///
/// # References:
///
/// \[0\] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct GaussNewtonLS<L, F> {
    /// linesearch
    linesearch: L,
    /// Tolerance for the stopping criterion based on cost difference
    tol: F,
}

impl<L, F: ArgminFloat> GaussNewtonLS<L, F> {
    /// Constructor
    pub fn new(linesearch: L) -> Self {
        GaussNewtonLS {
            linesearch,
            tol: F::epsilon().sqrt(),
        }
    }

    /// Set tolerance for the stopping criterion based on cost difference
    pub fn with_tol(mut self, tol: F) -> Result<Self, Error> {
        if tol <= F::from_f64(0.0).unwrap() {
            return Err(ArgminError::InvalidParameter {
                text: "Gauss-Newton-Linesearch: tol must be positive.".to_string(),
            }
            .into());
        }
        self.tol = tol;
        Ok(self)
    }
}

impl<O, L, F, P, G, J, U> Solver<O, IterState<P, G, J, (), F>> for GaussNewtonLS<L, F>
where
    O: Operator<Param = P, Output = U> + Jacobian<Param = P, Jacobian = J>,
    P: Clone + SerializeAlias + DeserializeOwnedAlias + ArgminMul<F, P>,
    G: Clone + SerializeAlias + DeserializeOwnedAlias,
    U: ArgminNorm<F>,
    J: Clone
        + SerializeAlias
        + DeserializeOwnedAlias
        + ArgminTranspose<J>
        + ArgminInv<J>
        + ArgminDot<J, J>
        + ArgminDot<G, P>
        + ArgminDot<U, G>,
    L: Clone + LineSearch<P, F> + Solver<LineSearchProblem<O, F>, IterState<P, G, (), (), F>>,
    F: ArgminFloat,
{
    const NAME: &'static str = "Gauss-Newton method with Linesearch";

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        mut state: IterState<P, G, J, (), F>,
    ) -> Result<(IterState<P, G, J, (), F>, Option<KV>), Error> {
        let param = state.take_param().unwrap();
        let residuals = problem.apply(&param)?;
        let jacobian = problem.jacobian(&param)?;
        let jacobian_t = jacobian.clone().t();
        let grad = jacobian_t.dot(&residuals);

        let p: P = jacobian_t.dot(&jacobian).inv()?.dot(&grad);

        self.linesearch
            .set_search_direction(p.mul(&(F::from_f64(-1.0).unwrap())));

        // perform linesearch
        let OptimizationResult {
            operator: mut line_problem,
            state: mut linesearch_state,
        } = Executor::new(
            LineSearchProblem::new(problem.take_problem().unwrap()),
            self.linesearch.clone(),
        )
        .configure(|config| config.param(param).grad(grad).cost(residuals.norm()))
        .ctrlc(false)
        .run()?;

        // Here we cannot use `consume_problem` because the problem we need is hidden inside a
        // `LineSearchProblem` hidden inside a `Problem`. Therefore we have to split this in two
        // separate tasks: first getting the operator, then dealing with the function counts.
        problem.problem = Some(line_problem.take_problem().unwrap().problem);
        problem.consume_func_counts(line_problem);

        Ok((
            state
                .param(linesearch_state.take_param().unwrap())
                .cost(linesearch_state.get_cost()),
            None,
        ))
    }

    fn terminate(&mut self, state: &IterState<P, G, J, (), F>) -> TerminationReason {
        if (state.get_prev_cost() - state.get_cost()).abs() < self.tol {
            return TerminationReason::NoChangeInCost;
        }
        TerminationReason::NotTerminated
    }
}

#[doc(hidden)]
#[derive(Clone, Default)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct LineSearchProblem<O, F> {
    pub problem: O,
    _phantom: std::marker::PhantomData<F>,
}

impl<O, F> LineSearchProblem<O, F> {
    /// constructor
    pub fn new(operator: O) -> Self {
        LineSearchProblem {
            problem: operator,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<O, P, F> CostFunction for LineSearchProblem<O, F>
where
    O: Operator<Param = P, Output = P>,
    P: Clone + SerializeAlias + DeserializeOwnedAlias + ArgminNorm<F>,
    F: ArgminFloat,
{
    type Param = P;
    type Output = F;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        Ok(self.problem.apply(p)?.norm())
    }
}

impl<O, P, J, F> Gradient for LineSearchProblem<O, F>
where
    O: Operator<Param = P, Output = P> + Jacobian<Param = P, Jacobian = J>,
    P: Clone + SerializeAlias + DeserializeOwnedAlias,
    J: ArgminTranspose<J> + ArgminDot<P, P>,
{
    type Param = P;
    type Gradient = P;

    fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
        Ok(self.problem.jacobian(p)?.t().dot(&self.problem.apply(p)?))
    }
}

impl<O, P, H, F> Hessian for LineSearchProblem<O, F>
where
    P: Clone + SerializeAlias + DeserializeOwnedAlias,
    H: Clone + SerializeAlias + DeserializeOwnedAlias,
    O: Hessian<Param = P, Hessian = H>,
{
    type Param = P;
    type Hessian = H;

    fn hessian(&self, p: &Self::Param) -> Result<Self::Hessian, Error> {
        self.problem.hessian(p)
    }
}

impl<O, P, J, F> Jacobian for LineSearchProblem<O, F>
where
    O: Jacobian<Param = P, Jacobian = J>,
    P: Clone + SerializeAlias + DeserializeOwnedAlias,
    J: Clone + SerializeAlias + DeserializeOwnedAlias,
{
    type Param = P;
    type Jacobian = J;

    fn jacobian(&self, p: &Self::Param) -> Result<Self::Jacobian, Error> {
        self.problem.jacobian(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::linesearch::MoreThuenteLineSearch;
    use crate::test_trait_impl;

    test_trait_impl!(
        gauss_newton_linesearch_method,
        GaussNewtonLS<MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64>, f64>
    );

    #[test]
    fn test_tolerance() {
        let tol1: f64 = 1e-4;

        let linesearch: MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64> =
            MoreThuenteLineSearch::new();
        let GaussNewtonLS { tol: t1, .. } = GaussNewtonLS::new(linesearch).with_tol(tol1).unwrap();

        assert!((t1 - tol1).abs() < std::f64::EPSILON);
    }
}
