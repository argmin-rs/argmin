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

/// Gauss-Newton method with linesearch
///
/// [Example](https://github.com/argmin-rs/argmin/blob/master/examples/gaussnewton_linesearch.rs)
///
/// # References:
///
/// [0] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(Clone, Serialize, Deserialize)]
pub struct GaussNewtonLS<L> {
    /// linesearch
    linesearch: L,
}

impl<L> GaussNewtonLS<L> {
    /// Constructor
    pub fn new(linesearch: L) -> Self {
        GaussNewtonLS { linesearch }
    }
}

impl<O, L> Solver<O> for GaussNewtonLS<L>
where
    O: ArgminOp,
    O::Param: Default
        + std::fmt::Debug
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
    L: Clone + ArgminLineSearch<O::Param> + Solver<OpWrapper<LineSearchOP<O>>>,
{
    const NAME: &'static str = "Gauss-Newton method with Linesearch";

    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &IterState<O>,
    ) -> Result<ArgminIterData<O>, Error> {
        let param = state.get_param();
        let residuals = op.apply(&param)?;
        let jacobian = op.jacobian(&param)?;
        let jacobian_t = jacobian.clone().t();
        let grad = jacobian_t.dot(&residuals);

        let p = jacobian_t.dot(&jacobian).inv()?.dot(&grad);

        self.linesearch.set_search_direction(p.mul(&(-1.0)));

        // create operator for linesearch
        let line_op = OpWrapper::new_move(LineSearchOP { op: op.clone_op() });

        // perform linesearch
        let ArgminResult {
            operator: line_op,
            state:
                IterState {
                    param: next_param,
                    cost: next_cost,
                    ..
                },
        } = Executor::new(line_op, self.linesearch.clone(), param)
            .grad(grad)
            .cost(residuals.norm())
            .ctrlc(false)
            .run()?;

        op.consume_op(line_op);

        Ok(ArgminIterData::new().param(next_param).cost(next_cost))
    }

    fn terminate(&mut self, state: &IterState<O>) -> TerminationReason {
        if (state.get_prev_cost() - state.get_cost()).abs() < std::f64::EPSILON.sqrt() {
            return TerminationReason::NoChangeInCost;
        }
        TerminationReason::NotTerminated
    }
}

#[doc(hidden)]
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct LineSearchOP<O> {
    op: O,
}

impl<O: ArgminOp> ArgminOp for LineSearchOP<O>
where
    O::Jacobian: ArgminTranspose + ArgminDot<O::Output, O::Param>,
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
    use crate::test_trait_impl;
    use crate::solver::linesearch::MoreThuenteLineSearch;

    test_trait_impl!(
        gauss_newton_linesearch_method,
        GaussNewtonLS<MoreThuenteLineSearch<Vec<f64>>>
    );
}
