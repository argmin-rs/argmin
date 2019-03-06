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
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// The Cauchy point is the minimum of the quadratic approximation of the cost function within the
/// trust region along the direction given by the first derivative.
///
/// # References:
///
/// [0] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(Clone, Serialize, Deserialize, Debug, Copy, PartialEq, PartialOrd, Default)]
pub struct CauchyPoint {
    /// Radius
    radius: f64,
}

impl CauchyPoint {
    /// Constructor
    ///
    /// Parameters:
    ///
    /// `operator`: operator
    pub fn new() -> Self {
        CauchyPoint {
            radius: std::f64::NAN,
        }
    }
}

impl<O> Solver<O> for CauchyPoint
where
    O: ArgminOp<Output = f64>,
    O::Param: Debug
        + Clone
        + Serialize
        + ArgminMul<f64, O::Param>
        + ArgminWeightedDot<O::Param, f64, O::Hessian>
        + ArgminNorm<f64>,
    O::Hessian: Clone + Serialize,
{
    fn next_iter(
        &mut self,
        _op: &mut OpWrapper<O>,
        state: IterState<O::Param, O::Hessian>,
    ) -> Result<ArgminIterData<O>, Error> {
        let grad_norm = state.cur_grad.norm();
        let wdp = state
            .cur_grad
            .weighted_dot(&state.cur_hessian, &state.cur_grad);
        let tau: f64 = if wdp <= 0.0 {
            1.0
        } else {
            1.0f64.min(grad_norm.powi(3) / (self.radius * wdp))
        };

        let new_param = state.cur_grad.mul(&(-tau * self.radius / grad_norm));
        Ok(ArgminIterData::new().param(new_param))
    }

    fn terminate(&mut self, state: &IterState<O::Param, O::Hessian>) -> TerminationReason {
        if state.cur_iter >= 1 {
            TerminationReason::MaxItersReached
        } else {
            TerminationReason::NotTerminated
        }
    }
}

impl ArgminTrustRegion for CauchyPoint {
    fn set_radius(&mut self, radius: f64) {
        self.radius = radius;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::send_sync_test;

    send_sync_test!(cauchypoint, CauchyPoint<MinimalNoOperator>);
}