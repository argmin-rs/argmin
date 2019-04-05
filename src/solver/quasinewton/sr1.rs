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
use std::fmt::Debug;

/// SR1 method (broken!)
///
/// [Example](https://github.com/argmin-rs/argmin/blob/master/examples/sr1.rs)
///
/// # References:
///
/// [0] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(Serialize, Deserialize)]
pub struct SR1<L, H> {
    /// parameter for skipping rule
    r: f64,
    /// Inverse Hessian
    inv_hessian: H,
    /// line search
    linesearch: L,
}

impl<L, H> SR1<L, H> {
    /// Constructor
    pub fn new(init_inverse_hessian: H, linesearch: L) -> Self {
        SR1 {
            r: 1e-8,
            inv_hessian: init_inverse_hessian,
            linesearch,
        }
    }

    /// Set r
    pub fn r(mut self, r: f64) -> Result<Self, Error> {
        if r < 0.0 || r > 1.0 {
            Err(ArgminError::InvalidParameter {
                text: "SR1: r must be between 0 and 1.".to_string(),
            }
            .into())
        } else {
            self.r = r;
            Ok(self)
        }
    }
}

impl<O, L, H> Solver<O> for SR1<L, H>
where
    O: ArgminOp<Output = f64, Hessian = H>,
    O::Param: Debug
        + Clone
        + Default
        + Serialize
        + ArgminSub<O::Param, O::Param>
        + ArgminDot<O::Param, f64>
        + ArgminDot<O::Param, O::Hessian>
        + ArgminNorm<f64>
        + ArgminMul<f64, O::Param>,
    O::Hessian: Debug
        + Clone
        + Default
        + Serialize
        + DeserializeOwned
        + ArgminSub<O::Hessian, O::Hessian>
        + ArgminDot<O::Param, O::Param>
        + ArgminDot<O::Hessian, O::Hessian>
        + ArgminAdd<O::Hessian, O::Hessian>
        + ArgminMul<f64, O::Hessian>,
    L: Clone + ArgminLineSearch<O::Param> + Solver<OpWrapper<O>>,
{
    const NAME: &'static str = "SR1";

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

        // Run solver
        let ArgminResult {
            operator: line_op,
            state:
                IterState {
                    param: xk1,
                    cost: next_cost,
                    ..
                },
        } = Executor::new(
            OpWrapper::new_from_op(&op),
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

        let skmhkyk: O::Param = sk.sub(&self.inv_hessian.dot(&yk));
        let a: O::Hessian = skmhkyk.dot(&skmhkyk);
        let b: f64 = skmhkyk.dot(&yk);

        let hessian_update = b.abs() >= self.r * yk.norm() * skmhkyk.norm();

        // a try to see whether the skipping rule based on B_k makes any difference (seems not)
        // let bk = self.inv_hessian.inv()?;
        // let ykmbksk = yk.sub(&bk.dot(&sk));
        // let tmp: f64 = sk.dot(&ykmbksk);
        // let sksk: f64 = sk.dot(&sk);
        // let blah: f64 = ykmbksk.dot(&ykmbksk);
        // let hessian_update = tmp.abs() >= self.r * sksk.sqrt() * blah.sqrt();

        if hessian_update {
            self.inv_hessian = self.inv_hessian.add(&a.mul(&(1.0 / b)));
        }

        Ok(ArgminIterData::new()
            .param(xk1)
            .cost(next_cost)
            .grad(grad)
            .kv(make_kv!["denom" => b; "hessian_update" => hessian_update;]))
    }

    fn terminate(&mut self, state: &IterState<O>) -> TerminationReason {
        if state.get_grad().unwrap().norm() < std::f64::EPSILON.sqrt() {
            return TerminationReason::TargetPrecisionReached;
        }
        if (state.get_prev_cost() - state.get_cost()).abs() < std::f64::EPSILON {
            return TerminationReason::NoChangeInCost;
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

    send_sync_test!(sr1, SR1<Operator, MoreThuenteLineSearch<Operator>>);
}
