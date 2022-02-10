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
    ArgminFloat, ArgminIterData, ArgminOp, ArgminTrustRegion, Error, IterState, OpWrapper, Solver,
    TerminationReason,
};
use argmin_math::{ArgminMul, ArgminNorm, ArgminWeightedDot};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// The Cauchy point is the minimum of the quadratic approximation of the cost function within the
/// trust region along the direction given by the first derivative.
///
/// # References:
///
/// \[0\] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(Clone, Debug, Copy, PartialEq, PartialOrd, Default)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct CauchyPoint<F> {
    /// Radius
    radius: F,
}

impl<F> CauchyPoint<F>
where
    F: ArgminFloat,
{
    /// Constructor
    pub fn new() -> Self {
        CauchyPoint { radius: F::nan() }
    }
}

impl<O, F> Solver<O> for CauchyPoint<F>
where
    O: ArgminOp<Output = F, Float = F>,
    O::Param: ArgminMul<O::Float, O::Param>
        + ArgminWeightedDot<O::Param, F, O::Hessian>
        + ArgminNorm<O::Float>,
    F: ArgminFloat,
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
        let tau: F = if wdp <= F::from_f64(0.0).unwrap() {
            F::from_f64(1.0).unwrap()
        } else {
            F::from_f64(1.0)
                .unwrap()
                .min(grad_norm.powi(3) / (self.radius * wdp))
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

impl<F> ArgminTrustRegion<F> for CauchyPoint<F>
where
    F: ArgminFloat,
{
    fn set_radius(&mut self, radius: F) {
        self.radius = radius;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_trait_impl;

    test_trait_impl!(cauchypoint, CauchyPoint<f64>);
}
