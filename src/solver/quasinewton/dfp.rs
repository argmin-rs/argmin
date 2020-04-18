// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # References:
//!
//! [0] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
//! Springer. ISBN 0-387-30303-0.

use crate::prelude::*;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

/// DFP method
///
/// [Example](https://github.com/argmin-rs/argmin/blob/master/examples/dfp.rs)
///
/// # References:
///
/// [0] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(Clone, Serialize, Deserialize)]
pub struct DFP<L, H, F> {
    /// Inverse Hessian
    inv_hessian: H,
    /// line search
    linesearch: L,
    /// Tolerance for the stopping criterion based on the change of the norm on the gradient
    tol_grad: F,
}

impl<L, H, F: ArgminFloat> DFP<L, H, F> {
    /// Constructor
    pub fn new(init_inverse_hessian: H, linesearch: L) -> Self {
        DFP {
            inv_hessian: init_inverse_hessian,
            linesearch,
            tol_grad: F::epsilon().sqrt(),
        }
    }

    /// Sets tolerance for the stopping criterion based on the change of the norm on the gradient
    pub fn with_tol_grad(mut self, tol_grad: F) -> Self {
        self.tol_grad = tol_grad;
        self
    }
}

impl<O, L, H, F> Solver<O, F> for DFP<L, H, F>
where
    O: ArgminOp<Output = F, Hessian = H>,
    O::Param: Clone
        + Default
        + Serialize
        + ArgminSub<O::Param, O::Param>
        + ArgminDot<O::Param, F>
        + ArgminDot<O::Param, O::Hessian>
        + ArgminScaledAdd<O::Param, F, O::Param>
        + ArgminNorm<F>
        + ArgminMul<F, O::Param>
        + ArgminTranspose,
    O::Hessian: Clone
        + Default
        + Serialize
        + DeserializeOwned
        + ArgminSub<O::Hessian, O::Hessian>
        + ArgminDot<O::Param, O::Param>
        + ArgminDot<O::Hessian, O::Hessian>
        + ArgminAdd<O::Hessian, O::Hessian>
        + ArgminMul<F, O::Hessian>
        + ArgminTranspose
        + ArgminEye,
    L: Clone + ArgminLineSearch<O::Param, F> + Solver<OpWrapper<O>, F>,
    F: ArgminFloat,
{
    const NAME: &'static str = "DFP";

    fn init(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &IterState<O, F>,
    ) -> Result<Option<ArgminIterData<O, F>>, Error> {
        let param = state.get_param();
        let cost = op.apply(&param)?;
        let grad = op.gradient(&param)?;
        Ok(Some(
            ArgminIterData::new().param(param).cost(cost).grad(grad),
        ))
    }

    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &IterState<O, F>,
    ) -> Result<ArgminIterData<O, F>, Error> {
        let param = state.get_param();
        let cost = state.get_cost();
        let prev_grad = if let Some(grad) = state.get_grad() {
            grad
        } else {
            op.gradient(&param)?
        };
        let p = self
            .inv_hessian
            .dot(&prev_grad)
            .mul(&F::from_f64(-1.0).unwrap());

        self.linesearch.set_search_direction(p);

        let ArgminResult {
            operator: line_op,
            state:
                IterState {
                    param: xk1,
                    cost: next_cost,
                    ..
                },
        } = Executor::new(
            OpWrapper::new_from_wrapper(op),
            self.linesearch.clone(),
            param.clone(),
        )
        .grad(prev_grad.clone())
        .cost(cost)
        .ctrlc(false)
        .run()?;

        // take care of function eval counts
        op.consume_op(line_op);

        let grad = op.gradient(&xk1)?;
        let yk = grad.sub(&prev_grad);

        let sk = xk1.sub(&param);

        let yksk: F = yk.dot(&sk);

        let sksk: O::Hessian = sk.dot(&sk);

        let tmp3: O::Param = self.inv_hessian.dot(&yk);
        let tmp4: F = tmp3.dot(&yk);
        let tmp3: O::Hessian = tmp3.dot(&tmp3);
        let tmp3: O::Hessian = tmp3.mul(&(F::from_f64(1.0).unwrap() / tmp4));

        self.inv_hessian = self
            .inv_hessian
            .sub(&tmp3)
            .add(&sksk.mul(&(F::from_f64(1.0).unwrap() / yksk)));

        Ok(ArgminIterData::new().param(xk1).cost(next_cost).grad(grad))
    }

    fn terminate(&mut self, state: &IterState<O, F>) -> TerminationReason {
        if state.get_grad().unwrap().norm() < self.tol_grad {
            return TerminationReason::TargetPrecisionReached;
        }
        TerminationReason::NotTerminated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
