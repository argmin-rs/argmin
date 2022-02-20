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
    ArgminError, ArgminFloat, ArgminKV, ArgminLineSearch, ArgminOp, DeserializeOwnedAlias, Error,
    Executor, Gradient, Hessian, IterState, OpWrapper, Operator, OptimizationResult,
    SerializeAlias, Solver, State, TerminationReason,
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

impl<O, L, P, H, F> Solver<O, IterState<O>> for NewtonCG<L, F>
where
    O: ArgminOp<Param = P, Hessian = H, Output = F, Float = F>
        + Gradient<Param = P, Gradient = P, Float = F>
        + Hessian<Param = P, Hessian = H, Float = F>,
    P: Clone
        + SerializeAlias
        + DeserializeOwnedAlias
        + ArgminSub<P, P>
        + ArgminDot<P, F>
        + ArgminScaledAdd<P, F, P>
        + ArgminMul<F, P>
        + ArgminConj
        + ArgminZeroLike
        + ArgminNorm<F>,
    H: Clone + SerializeAlias + DeserializeOwnedAlias + ArgminDot<P, P>,
    L: Clone + ArgminLineSearch<P, F> + Solver<O, IterState<O>>,
    F: ArgminFloat + ArgminNorm<F>,
{
    const NAME: &'static str = "Newton-CG";

    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        mut state: IterState<O>,
    ) -> Result<(IterState<O>, Option<ArgminKV>), Error> {
        let param = state.take_param().unwrap();
        let grad = op.gradient(&param)?;
        let hessian = op.hessian(&param)?;

        // Solve CG subproblem
        let cg_op: CGSubProblem<P, H, F> = CGSubProblem::new(hessian.clone());
        let mut cg_op = OpWrapper::new(cg_op);

        let mut x_p = param.zero_like();
        let mut x: P = param.zero_like();
        let mut cg = ConjugateGradient::new(grad.mul(&(F::from_f64(-1.0).unwrap())))?;

        let cg_state = IterState::new().param(x_p.clone());
        let (mut cg_state, _) = cg.init(&mut cg_op, cg_state)?;
        let grad_norm = grad.norm();
        for iter in 0.. {
            let (state_tmp, _) = cg.next_iter(&mut cg_op, cg_state)?;
            cg_state = state_tmp;
            let cost = cg_state.get_cost();
            x = cg_state.take_param().unwrap();
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
        let OptimizationResult {
            operator: line_op,
            state: mut linesearch_state,
        } = Executor::new(op.take_op().unwrap(), self.linesearch.clone())
            .configure(|config| config.param(param).grad(grad).cost(line_cost))
            .ctrlc(false)
            .run()?;

        op.consume_op(line_op);

        Ok((
            state
                .param(linesearch_state.take_param().unwrap())
                .cost(linesearch_state.get_cost()),
            None,
        ))
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

impl<P, H, F> ArgminOp for CGSubProblem<P, H, F>
where
    P: Clone + SerializeAlias + DeserializeOwnedAlias,
    H: Clone + ArgminDot<P, P> + SerializeAlias + DeserializeOwnedAlias,
    F: ArgminFloat,
{
    type Param = P;
    type Output = P;
    type Hessian = ();
    type Jacobian = ();
    type Float = F;
}

impl<P, H, F> Operator for CGSubProblem<P, H, F>
where
    H: ArgminDot<P, P>,
    F: ArgminFloat,
{
    type Param = P;
    type Output = P;
    type Float = F;

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
