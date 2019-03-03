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
use std::default::Default;

/// The Cauchy point is the minimum of the quadratic approximation of the cost function within the
/// trust region along the direction given by the first derivative.
///
/// # References:
///
/// [0] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(Serialize, Deserialize)]
pub struct CauchyPoint<P, H> {
    /// Radius
    radius: f64,
    /// Gradient
    grad: P,
    /// Hessian
    hessian: H,
}

impl<P, H> CauchyPoint<P, H>
where
    P: Default,
    H: Default,
{
    /// Constructor
    ///
    /// Parameters:
    ///
    /// `operator`: operator
    pub fn new() -> Self {
        CauchyPoint {
            radius: std::f64::NAN,
            grad: P::default(),
            hessian: H::default(),
        }
    }
}

impl<O, P, H> Solver<O> for CauchyPoint<P, H>
where
    O: ArgminOp<Param = P, Output = f64, Hessian = H>,
    P: Clone + Serialize + ArgminMul<f64, P> + ArgminWeightedDot<P, f64, H> + ArgminNorm<f64>,
    H: Clone + Serialize,
{
    fn next_iter(
        &mut self,
        _op: &mut OpWrapper<O>,
        state: IterState<P, H>,
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

impl<P, H> ArgminTrustRegion<P, H> for CauchyPoint<P, H>
where
    P: Serialize,
    H: Serialize,
{
    fn set_radius(&mut self, radius: f64) {
        self.radius = radius;
    }

    fn set_grad(&mut self, grad: P) {
        self.grad = grad;
    }

    fn set_hessian(&mut self, hessian: H) {
        self.hessian = hessian;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::send_sync_test;

    send_sync_test!(cauchypoint, CauchyPoint<MinimalNoOperator>);
}
