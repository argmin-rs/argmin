// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! TODO: Stop when search direction is close to 0
//!
//! # References:
//!
//! \[0\] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
//! Springer. ISBN 0-387-30303-0.

use crate::core::{
    ArgminError, ArgminFloat, ArgminIterData, ArgminLineSearch, ArgminOp, ArgminResult,
    DeserializeOwnedAlias, Error, Executor, IterState, OpWrapper, SerializeAlias, Solver, State,
    TerminationReason,
};
use crate::solver::conjugategradient::ConjugateGradient;
use argmin_math::{
    ArgminConj, ArgminDot, ArgminMul, ArgminNorm, ArgminScaledAdd, ArgminSub, ArgminZeroLike,
};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// The Newton-CG method (also called truncated Newton method) uses a modified CG to solve the
/// Newton equations approximately. After a search direction is found, a line search is performed.
///
/// # References:
///
/// \[0\] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
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
            curvature_threshold: F::from_f64(0.0).unwrap(),
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
    pub fn with_tol(mut self, tol: F) -> Result<Self, Error> {
        if tol <= F::from_f64(0.0).unwrap() {
            return Err(ArgminError::InvalidParameter {
                text: "Newton-CG: tol must be positive.".to_string(),
            }
            .into());
        }
        self.tol = tol;
        Ok(self)
    }
}

impl<O, L, F> Solver<IterState<O>> for NewtonCG<L, F>
where
    O: ArgminOp<Output = F, Float = F>,
    O::Param: SerializeAlias
        + ArgminSub<O::Param, O::Param>
        + ArgminDot<O::Param, O::Float>
        + ArgminScaledAdd<O::Param, O::Float, O::Param>
        + ArgminMul<F, O::Param>
        + ArgminConj
        + ArgminZeroLike
        + ArgminNorm<O::Float>,
    O::Hessian: ArgminDot<O::Param, O::Param>,
    L: Clone + ArgminLineSearch<O::Param, O::Float> + Solver<IterState<O>>,
    F: ArgminFloat + ArgminNorm<O::Float>,
{
    const NAME: &'static str = "Newton-CG";

    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &mut IterState<O>,
    ) -> Result<ArgminIterData<IterState<O>>, Error> {
        let param = state.take_param().unwrap();
        let grad = op.gradient(&param)?;
        let hessian = op.hessian(&param)?;

        // Solve CG subproblem
        let cg_op: CGSubProblem<O::Param, O::Hessian, O::Float> =
            CGSubProblem::new(hessian.clone());
        let mut cg_op = OpWrapper::new(cg_op);

        let mut x_p = param.zero_like();
        let mut x: O::Param = param.zero_like();
        let mut cg = ConjugateGradient::new(grad.mul(&(F::from_f64(-1.0).unwrap())))?;

        let mut cg_state = IterState::new().param(x_p.clone());
        cg.init(&mut cg_op, &mut cg_state)?;
        let grad_norm = grad.norm();
        for iter in 0.. {
            let data = cg.next_iter(&mut cg_op, &mut cg_state)?;
            let cost = data.get_cost().unwrap();
            x = data.get_param().unwrap();
            let p = cg.p_prev();
            let curvature = p.dot(&hessian.dot(p));
            if curvature <= self.curvature_threshold {
                if iter == 0 {
                    x = grad.mul(&(F::from_f64(-1.0).unwrap()));
                } else {
                    x = x_p;
                }
                break;
            }
            if cost <= F::from_f64(0.5).unwrap().min(grad_norm.sqrt()) * grad_norm {
                break;
            }
            cg_state = cg_state.param(x.clone()).cost(cost);
            x_p = x.clone();
        }

        // perform line search
        self.linesearch.set_search_direction(x);

        let line_cost = state.get_cost();

        // Run solver
        let ArgminResult {
            operator: line_op,
            state: mut linesearch_state,
        } = Executor::new(op.take_op().unwrap(), self.linesearch.clone())
            .configure(|config| config.param(param).grad(grad).cost(line_cost))
            .ctrlc(false)
            .run()?;

        op.consume_op(line_op);

        Ok(ArgminIterData::new()
            .param(linesearch_state.take_param().unwrap())
            .cost(linesearch_state.get_cost()))
    }

    fn terminate(&mut self, state: &IterState<O>) -> TerminationReason {
        if (state.get_cost() - state.get_prev_cost()).abs() < self.tol {
            TerminationReason::NoChangeInCost
        } else {
            TerminationReason::NotTerminated
        }
    }
}

#[derive(Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
struct CGSubProblem<T, H, F> {
    hessian: H,
    phantom: std::marker::PhantomData<T>,
    float: std::marker::PhantomData<F>,
}

impl<T, H, F> CGSubProblem<T, H, F> {
    /// constructor
    pub fn new(hessian: H) -> Self {
        CGSubProblem {
            hessian,
            phantom: std::marker::PhantomData,
            float: std::marker::PhantomData,
        }
    }
}

impl<T, H, F> ArgminOp for CGSubProblem<T, H, F>
where
    T: Clone + SerializeAlias + DeserializeOwnedAlias,
    H: Clone + ArgminDot<T, T> + SerializeAlias + DeserializeOwnedAlias,
    F: ArgminFloat,
{
    type Param = T;
    type Output = T;
    type Hessian = ();
    type Jacobian = ();
    type Float = F;

    fn apply(&self, p: &T) -> Result<T, Error> {
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
        NewtonCG<MoreThuenteLineSearch<Vec<f64>, f64>, f64>
    );

    test_trait_impl!(cg_subproblem, CGSubProblem<Vec<f64>, Vec<Vec<f64>>, f64>);

    #[test]
    fn test_tolerance() {
        let tol1: f64 = 1e-4;

        let linesearch: MoreThuenteLineSearch<Vec<f64>, f64> = MoreThuenteLineSearch::new();

        let NewtonCG { tol: t, .. }: NewtonCG<MoreThuenteLineSearch<Vec<f64>, f64>, f64> =
            NewtonCG::new(linesearch).with_tol(tol1).unwrap();

        assert!((t - tol1).abs() < std::f64::EPSILON);
    }
}
