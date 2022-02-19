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
    ArgminFloat, ArgminIterData, ArgminLineSearch, ArgminOp, ArgminResult, Error, Executor,
    IterState, OpWrapper, SerializeAlias, Solver, State, TerminationReason,
};
use argmin_math::{ArgminAdd, ArgminDot, ArgminMul, ArgminNorm, ArgminSub};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// DFP method
///
/// # References:
///
/// \[0\] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct DFP<L, H, F> {
    /// Initial inverse Hessian
    init_inv_hessian: H,
    /// line search
    linesearch: L,
    /// Tolerance for the stopping criterion based on the change of the norm on the gradient
    tol_grad: F,
}

impl<L, H, F> DFP<L, H, F>
where
    F: ArgminFloat,
{
    /// Constructor
    pub fn new(init_inverse_hessian: H, linesearch: L) -> Self {
        DFP {
            init_inv_hessian: init_inverse_hessian,
            linesearch,
            tol_grad: F::epsilon().sqrt(),
        }
    }

    /// Sets tolerance for the stopping criterion based on the change of the norm on the gradient
    #[must_use]
    pub fn with_tol_grad(mut self, tol_grad: F) -> Self {
        self.tol_grad = tol_grad;
        self
    }
}

impl<O, L, H, F> Solver<IterState<O>> for DFP<L, H, F>
where
    O: ArgminOp<Output = F, Hessian = H, Float = F>,
    O::Param: ArgminSub<O::Param, O::Param>
        + ArgminDot<O::Param, O::Float>
        + ArgminDot<O::Param, O::Hessian>
        + ArgminNorm<O::Float>
        + ArgminMul<O::Float, O::Param>,
    H: Clone
        + SerializeAlias
        + ArgminSub<O::Hessian, O::Hessian>
        + ArgminDot<O::Param, O::Param>
        + ArgminAdd<O::Hessian, O::Hessian>
        + ArgminMul<F, O::Hessian>,
    L: Clone + ArgminLineSearch<O::Param, O::Float> + Solver<IterState<O>>,
    F: ArgminFloat,
{
    const NAME: &'static str = "DFP";

    fn init(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &mut IterState<O>,
    ) -> Result<Option<ArgminIterData<IterState<O>>>, Error> {
        let param = state.take_param().unwrap();
        let cost = op.apply(&param)?;
        let grad = op.gradient(&param)?;
        Ok(Some(
            ArgminIterData::new()
                .param(param)
                .cost(cost)
                .grad(grad)
                .inv_hessian(self.init_inv_hessian.clone()),
        ))
    }

    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &mut IterState<O>,
    ) -> Result<ArgminIterData<IterState<O>>, Error> {
        let param = state.take_param().unwrap();
        let cost = state.get_cost();
        let prev_grad = state
            .take_grad()
            .map(Result::Ok)
            .unwrap_or_else(|| op.gradient(&param))?;
        let inv_hessian = state.get_inv_hessian().unwrap();
        let p = inv_hessian.dot(&prev_grad).mul(&F::from_f64(-1.0).unwrap());

        self.linesearch.set_search_direction(p);

        let ArgminResult {
            operator: line_op,
            state: mut linesearch_state,
        } = Executor::new(op.take_op().unwrap(), self.linesearch.clone())
            .configure(|config| {
                config
                    .param(param.clone())
                    .grad(prev_grad.clone())
                    .cost(cost)
            })
            .ctrlc(false)
            .run()?;

        let xk1 = linesearch_state.take_param().unwrap();
        let next_cost = linesearch_state.get_cost();

        // take care of function eval counts
        op.consume_op(line_op);

        let grad = op.gradient(&xk1)?;
        let yk = grad.sub(&prev_grad);

        let sk = xk1.sub(&param);

        let yksk: F = yk.dot(&sk);

        let sksk: O::Hessian = sk.dot(&sk);

        let tmp3: O::Param = inv_hessian.dot(&yk);
        let tmp4: F = tmp3.dot(&yk);
        let tmp3: O::Hessian = tmp3.dot(&tmp3);
        let tmp3: O::Hessian = tmp3.mul(&(F::from_f64(1.0).unwrap() / tmp4));

        let inv_hessian = inv_hessian
            .sub(&tmp3)
            .add(&sksk.mul(&(F::from_f64(1.0).unwrap() / yksk)));

        Ok(ArgminIterData::new()
            .param(xk1)
            .cost(next_cost)
            .grad(grad)
            .inv_hessian(inv_hessian))
    }

    fn terminate(&mut self, state: &IterState<O>) -> TerminationReason {
        if state.get_grad_ref().unwrap().norm() < self.tol_grad {
            return TerminationReason::TargetPrecisionReached;
        }
        TerminationReason::NotTerminated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::MinimalNoOperator;
    use crate::solver::linesearch::MoreThuenteLineSearch;
    use crate::test_trait_impl;

    type Operator = MinimalNoOperator;

    test_trait_impl!(dfp, DFP<Operator, MoreThuenteLineSearch<Operator, f64>, f64>);

    #[test]
    fn test_tolerances() {
        let linesearch: MoreThuenteLineSearch<f64, f64> =
            MoreThuenteLineSearch::new().c(1e-4, 0.9).unwrap();
        let init_hessian: Vec<Vec<f64>> = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        let tol: f64 = 1e-4;

        let DFP { tol_grad: t, .. } = DFP::new(init_hessian, linesearch).with_tol_grad(tol);

        assert!((t - tol).abs() < std::f64::EPSILON);
    }
}
