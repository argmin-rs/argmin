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
pub struct GaussNewton<L> {
    /// gamma
    gamma: f64,
    /// line search
    linesearch: L,
}

impl<L> GaussNewton<L> {
    /// Constructor
    pub fn new(linesearch: L) -> Self {
        GaussNewton {
            gamma: 1.0,
            linesearch,
        }
    }

    /// set gamma
    pub fn gamma(mut self, gamma: f64) -> Result<Self, Error> {
        if gamma <= 0.0 || gamma > 1.0 {
            return Err(ArgminError::InvalidParameter {
                text: "Gauss-Newton: gamma must be in  (0, 1].".to_string(),
            }
            .into());
        }
        self.gamma = gamma;
        Ok(self)
    }
}

// impl Default for GaussNewton {
//     fn default() -> GaussNewton {
//         GaussNewton::new()
//     }
// }

impl<O, L> Solver<O> for GaussNewton<L>
where
    O: ArgminOp,
    O::Param: Default
        + ArgminScaledSub<O::Param, f64, O::Param>
        + ArgminDot<O::Param, O::Param>
        + ArgminAdd<O::Param, O::Param>
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
    const NAME: &'static str = "Gauss-Newton method";

    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &IterState<O>,
    ) -> Result<ArgminIterData<O>, Error> {
        let param = state.get_param();
        let residuals = op.apply(&param)?;
        // let grad = op.gradient(&param)?;
        let jacobian = op.jacobian(&param)?;
        let jacobian_t = jacobian.clone().t();

        let p = jacobian_t
            .dot(&jacobian)
            .inv()?
            .dot(&jacobian.t().dot(&residuals));
        // .mul(&1.0);

        // self.linesearch.set_search_direction(p);

        // TODO: Need to build another operator which does not return the residuals when calling
        // `apply`, but instead returns the norm of the residuals. Otherwise this may not work....
        // let ArgminResult {
        //     operator: line_op,
        //     state:
        //         IterState {
        //             param: new_param,
        //             cost: new_cost,
        //             ..
        //         },
        // } = Executor::new(OpWrapper::new_from_op(&op), self.linesearch.clone(), param)
        //     .grad(grad)
        //     .cost(residuals.norm())
        //     .ctrlc(false)
        //     .run()?;
        // op.consume_op(line_op);
        // let new_param = param.scaled_sub(&self.gamma, &p.dot(&grad));

        let new_param = param.add(&p.dot(&param));

        Ok(ArgminIterData::new()
            .param(new_param)
            .cost(residuals.norm()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::send_sync_test;
    use crate::solver::linesearch::MoreThuenteLineSearch;

    send_sync_test!(
        gauss_newton_method,
        GaussNewton<MoreThuenteLineSearch<Vec<f64>>>
    );
}
