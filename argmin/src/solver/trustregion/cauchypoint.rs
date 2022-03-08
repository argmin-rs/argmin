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
    ArgminFloat, Error, Gradient, Hessian, IterState, OpWrapper, Solver, State, TerminationReason,
    TrustRegionRadius, KV,
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

impl<O, F, P, G, H> Solver<O, IterState<P, G, (), H, F>> for CauchyPoint<F>
where
    O: Gradient<Param = P, Gradient = G> + Hessian<Param = P, Hessian = H>,
    P: Clone + ArgminMul<F, P> + ArgminWeightedDot<P, F, H> + ArgminNorm<F>,
    G: ArgminMul<F, P> + ArgminWeightedDot<G, F, H> + ArgminNorm<F>,
    F: ArgminFloat,
{
    const NAME: &'static str = "Cauchy Point";

    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        mut state: IterState<P, G, (), H, F>,
    ) -> Result<(IterState<P, G, (), H, F>, Option<KV>), Error> {
        let param = state.take_param().unwrap();
        let grad = state
            .take_grad()
            .map(Result::Ok)
            .unwrap_or_else(|| op.gradient(&param))?;
        let grad_norm = grad.norm();
        let hessian = state
            .take_hessian()
            .map(Result::Ok)
            .unwrap_or_else(|| op.hessian(&param))?;

        let wdp = grad.weighted_dot(&hessian, &grad);
        let tau: F = if wdp <= F::from_f64(0.0).unwrap() {
            F::from_f64(1.0).unwrap()
        } else {
            F::from_f64(1.0)
                .unwrap()
                .min(grad_norm.powi(3) / (self.radius * wdp))
        };

        let new_param = grad.mul(&(-tau * self.radius / grad_norm));
        Ok((state.param(new_param).grad(grad).hessian(hessian), None))
    }

    fn terminate(&mut self, state: &IterState<P, G, (), H, F>) -> TerminationReason {
        if state.get_iter() >= 1 {
            TerminationReason::MaxItersReached
        } else {
            TerminationReason::NotTerminated
        }
    }
}

impl<F> TrustRegionRadius<F> for CauchyPoint<F>
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
