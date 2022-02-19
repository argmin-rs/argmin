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
    ArgminError, ArgminFloat, ArgminIterData, ArgminLineSearch, ArgminOp, ArgminResult, Error,
    Executor, IterState, OpWrapper, Solver, State, TerminationReason,
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

impl<O, L, F> Solver<IterState<O>> for GaussNewtonLS<L, F>
where
    O: ArgminOp<Float = F>,
    O::Param: ArgminMul<O::Float, O::Param>,
    O::Output: ArgminNorm<O::Float>,
    O::Jacobian: ArgminTranspose<O::Jacobian>
        + ArgminInv<O::Jacobian>
        + ArgminDot<O::Jacobian, O::Jacobian>
        + ArgminDot<O::Output, O::Param>
        + ArgminDot<O::Param, O::Param>,
    L: Clone + ArgminLineSearch<O::Param, O::Float> + Solver<IterState<LineSearchOP<O>>>,
    F: ArgminFloat,
{
    const NAME: &'static str = "Gauss-Newton method with Linesearch";

    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &mut IterState<O>,
    ) -> Result<ArgminIterData<IterState<O>>, Error> {
        let param = state.take_param().unwrap();
        let residuals = op.apply(&param)?;
        let jacobian = op.jacobian(&param)?;
        let jacobian_t = jacobian.clone().t();
        let grad = jacobian_t.dot(&residuals);

        let p = jacobian_t.dot(&jacobian).inv()?.dot(&grad);

        self.linesearch
            .set_search_direction(p.mul(&(F::from_f64(-1.0).unwrap())));

        // perform linesearch
        let ArgminResult {
            operator: mut line_op,
            state: mut linesearch_state,
        } = Executor::new(
            LineSearchOP {
                op: op.take_op().unwrap(),
            },
            self.linesearch.clone(),
            param,
        )
        .configure(|config| config.grad(grad).cost(residuals.norm()))
        .ctrlc(false)
        .run()?;

        // Here we cannot use `consume_op` because the operator we need is hidden inside a
        // `LineSearchOP` hidden inside a `OpWrapper`. Therefore we have to split this in two
        // separate tasks: first getting the operator, then dealing with the function counts.
        op.op = Some(line_op.take_op().unwrap().op);
        op.consume_func_counts(line_op);

        Ok(ArgminIterData::new()
            .param(linesearch_state.take_param().unwrap())
            .cost(linesearch_state.get_cost()))
    }

    fn terminate(&mut self, state: &IterState<O>) -> TerminationReason {
        if (state.get_prev_cost() - state.get_cost()).abs() < self.tol {
            return TerminationReason::NoChangeInCost;
        }
        TerminationReason::NotTerminated
    }
}

#[doc(hidden)]
#[derive(Clone, Default)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct LineSearchOP<O> {
    pub op: O,
}

impl<O> ArgminOp for LineSearchOP<O>
where
    O: ArgminOp,
    O::Jacobian: ArgminTranspose<O::Jacobian> + ArgminDot<O::Output, O::Param>,
    O::Output: ArgminNorm<O::Float>,
{
    type Param = O::Param;
    type Output = O::Float;
    type Hessian = O::Hessian;
    type Jacobian = O::Jacobian;
    type Float = O::Float;

    fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        Ok(self.op.apply(p)?.norm())
    }

    fn gradient(&self, p: &Self::Param) -> Result<Self::Param, Error> {
        Ok(self.op.jacobian(p)?.t().dot(&self.op.apply(p)?))
    }

    fn hessian(&self, p: &Self::Param) -> Result<Self::Hessian, Error> {
        self.op.hessian(p)
    }

    fn jacobian(&self, p: &Self::Param) -> Result<Self::Jacobian, Error> {
        self.op.jacobian(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::linesearch::MoreThuenteLineSearch;
    use crate::test_trait_impl;

    test_trait_impl!(
        gauss_newton_linesearch_method,
        GaussNewtonLS<MoreThuenteLineSearch<Vec<f64>, f64>, f64>
    );

    #[test]
    fn test_tolerance() {
        let tol1: f64 = 1e-4;

        let linesearch: MoreThuenteLineSearch<Vec<f64>, f64> = MoreThuenteLineSearch::new();
        let GaussNewtonLS { tol: t1, .. } = GaussNewtonLS::new(linesearch).with_tol(tol1).unwrap();

        assert!((t1 - tol1).abs() < std::f64::EPSILON);
    }
}
