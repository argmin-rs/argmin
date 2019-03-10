// Copyright 2018 Stefan Kroboth
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
/// # Example
///
/// ```rust
/// TODO
/// ```
///
/// # References:
///
/// [0] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(Serialize, Deserialize)]
pub struct DFP<L, H> {
    /// Inverse Hessian
    inv_hessian: H,
    /// line search
    linesearch: L,
}

impl<L, H> DFP<L, H> {
    /// Constructor
    pub fn new(init_inverse_hessian: H, linesearch: L) -> Self {
        DFP {
            inv_hessian: init_inverse_hessian,
            linesearch,
        }
    }
}

impl<O, L, H> Solver<O> for DFP<L, H>
where
    O: ArgminOp<Output = f64, Hessian = H>,
    O::Param: Clone
        + Default
        + Serialize
        + ArgminSub<O::Param, O::Param>
        + ArgminDot<O::Param, f64>
        + ArgminDot<O::Param, O::Hessian>
        + ArgminScaledAdd<O::Param, f64, O::Param>
        + ArgminNorm<f64>
        + ArgminMul<f64, O::Param>
        + ArgminTranspose,
    O::Hessian: Clone
        + Default
        + Serialize
        + DeserializeOwned
        + ArgminSub<O::Hessian, O::Hessian>
        + ArgminDot<O::Param, O::Param>
        + ArgminDot<O::Hessian, O::Hessian>
        + ArgminAdd<O::Hessian, O::Hessian>
        + ArgminMul<f64, O::Hessian>
        + ArgminTranspose
        + ArgminEye,
    L: Clone + ArgminLineSearch<O::Param> + Solver<OpWrapper<O>>,
{
    const NAME: &'static str = "DFP";

    fn init(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &IterState<O>,
    ) -> Result<Option<ArgminIterData<O>>, Error> {
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
        state: &IterState<O>,
    ) -> Result<ArgminIterData<O>, Error> {
        let param = state.get_param();
        let cost = state.get_cost();
        let prev_grad = if let Some(grad) = state.get_grad() {
            grad
        } else {
            op.gradient(&param)?
        };
        let p = self.inv_hessian.dot(&prev_grad).mul(&(-1.0));

        self.linesearch.set_search_direction(p);

        let linesearch_result = Executor::new(
            OpWrapper::new_from_op(&op),
            self.linesearch.clone(),
            param.clone(),
        )
        .grad(prev_grad.clone())
        .cost(cost)
        .run_fast()?;

        // take care of function eval counts
        op.consume_op(linesearch_result.operator);

        let xk1 = linesearch_result.param;

        let grad = op.gradient(&xk1)?;
        let yk = grad.sub(&prev_grad);

        let sk = xk1.sub(&param);

        let yksk: f64 = yk.dot(&sk);

        let sksk: O::Hessian = sk.dot(&sk);

        let tmp3: O::Param = self.inv_hessian.dot(&yk);
        let tmp4: f64 = tmp3.dot(&yk);
        let tmp3: O::Hessian = tmp3.dot(&tmp3);
        let tmp3: O::Hessian = tmp3.mul(&(1.0f64 / tmp4));

        self.inv_hessian = self.inv_hessian.sub(&tmp3).add(&sksk.mul(&(1.0f64 / yksk)));

        Ok(ArgminIterData::new()
            .param(xk1)
            .cost(linesearch_result.cost)
            .grad(grad))
    }

    fn terminate(&mut self, state: &IterState<O>) -> TerminationReason {
        if state.get_grad().unwrap().norm() < std::f64::EPSILON.sqrt() {
            return TerminationReason::TargetPrecisionReached;
        }
        TerminationReason::NotTerminated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::send_sync_test;
    use crate::solver::linesearch::MoreThuenteLineSearch;

    type Operator = MinimalNoOperator;

    send_sync_test!(dfp, DFP<Operator, MoreThuenteLineSearch<Operator>>);
}
