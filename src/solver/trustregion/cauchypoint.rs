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
    const NAME: &'static str = "Cauchy Point";

    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &IterState<O>,
    ) -> Result<ArgminIterData<O>, Error> {
        let param = state.get_param();
        let grad = state
            .get_grad()
            .unwrap_or_else(|| op.gradient(&param).unwrap());
        let grad_norm = grad.norm();
        let hessian = state
            .get_hessian()
            .unwrap_or_else(|| op.hessian(&param).unwrap());

        let wdp = grad.weighted_dot(&hessian, &grad);
        let tau: f64 = if wdp <= 0.0 {
            1.0
        } else {
            1.0f64.min(grad_norm.powi(3) / (self.radius * wdp))
        };

        let new_param = grad.mul(&(-tau * self.radius / grad_norm));
        Ok(ArgminIterData::new().param(new_param))
    }

    fn terminate(&mut self, state: &IterState<O>) -> TerminationReason {
        if state.get_iter() >= 1 {
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

    send_sync_test!(cauchypoint, CauchyPoint);
}
