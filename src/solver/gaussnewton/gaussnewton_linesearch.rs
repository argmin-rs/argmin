// Copyright 2018-2019 Stefan Kroboth
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
use serde::{Deserialize, Serialize};
use std::default::Default;

/// Gauss-Newton method (untested!)
///
/// [Example](https://github.com/argmin-rs/argmin/blob/master/examples/gaussnewton.rs)
///
/// # References:
///
/// [0] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(Serialize, Deserialize)]
pub struct GaussNewtonLinesearch<L> {
    /// linesearch
    linesearch: L,
}

impl<L> GaussNewtonLinesearch<L> {
    /// Constructor
    pub fn new(linesearch: L) -> Self {
        GaussNewtonLinesearch { linesearch }
    }
}

// impl Default for GaussNewton {
//     fn default() -> GaussNewton {
//         GaussNewton::new()
//     }
// }

impl<O, L> Solver<O> for GaussNewtonLinesearch<L>
where
    O: ArgminOp,
    O::Param: Default
        + ArgminScaledSub<O::Param, f64, O::Param>
        + ArgminSub<O::Param, O::Param>
        + ArgminMul<f64, O::Param>,
    O::Output: ArgminNorm<f64>,
    O::Jacobian: ArgminTranspose
        + ArgminInv<O::Jacobian>
        + ArgminDot<O::Jacobian, O::Jacobian>
        + ArgminDot<O::Output, O::Param>
        + ArgminDot<O::Param, O::Param>,
    O::Hessian: Default,
    L: Clone + ArgminLineSearch<O::Param> + Solver<OpWrapper<O>>,
{
    const NAME: &'static str = "Gauss-Newton method with Linesearch";

    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &IterState<O>,
    ) -> Result<ArgminIterData<O>, Error> {
        let param = state.get_param();
        let residuals = op.apply(&param)?;
        let grad = op.gradient(&param)?;
        let jacobian = op.jacobian(&param)?;
        let jacobian_t = jacobian.clone().t();

        let p = jacobian_t
            .dot(&jacobian)
            .inv()?
            .dot(&jacobian.t().dot(&residuals));

        self.linesearch.set_search_direction(p.mul(&(-1.0)));

        // create operator for linesearch

        // perform linesearch
        let ArgminResult {
            operator: line_op,
            state:
                IterState {
                    param: next_param,
                    cost: next_cost,
                    ..
                },
        } = Executor::new(OpWrapper::new_from_op(&op), self.linesearch.clone(), param)
            .grad(grad)
            .cost(residuals.norm())
            .ctrlc(false)
            .run()?;

        op.consume_op(line_op);
        // let new_param = param.sub(&p);

        Ok(ArgminIterData::new().param(next_param).cost(next_cost))
    }

    fn terminate(&mut self, state: &IterState<O>) -> TerminationReason {
        if (state.get_prev_cost() - state.get_cost()).abs() < std::f64::EPSILON.sqrt() {
            return TerminationReason::NoChangeInCost;
        }
        TerminationReason::NotTerminated
    }
}

#[derive(Clone, Default, Serialize, Deserialize)]
struct LinesearchOP<O> {
    op: O,
}

impl<O: ArgminOp> ArgminOp for LinesearchOP<O>
where
    O::Output: ArgminNorm<f64>,
{
    type Param = O::Param;
    type Output = f64;
    type Hessian = O::Hessian;
    type Jacobian = O::Jacobian;

    fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        Ok(self.op.apply(p)?.norm())
    }

    fn gradient(&self, p: &Self::Param) -> Result<Self::Param, Error> {
        self.op.gradient(p)
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
    use crate::send_sync_test;
    use crate::solver::linesearch::MoreThuenteLineSearch;

    send_sync_test!(
        gauss_newton_linesearch_method,
        GaussNewtonLinesearch<MoreThuenteLineSearch<Vec<f64>>>
    );
}
