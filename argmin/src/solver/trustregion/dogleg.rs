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

use crate::prelude::*;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// The Dogleg method computes the intersection of the trust region boundary with a path given by
/// the unconstraind minimum along the steepest descent direction and the optimum of the quadratic
/// approximation of the cost function at the current point.
///
/// # References:
///
/// \[0\] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(Clone, Debug, Copy, PartialEq, PartialOrd, Default)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct Dogleg<F> {
    /// Radius
    radius: F,
}

impl<F: ArgminFloat> Dogleg<F> {
    /// Constructor
    pub fn new() -> Self {
        Dogleg { radius: F::nan() }
    }
}

impl<O, F> Solver<O> for Dogleg<F>
where
    O: ArgminOp<Output = F, Float = F>,
    O::Param: std::fmt::Debug
        + ArgminMul<F, O::Param>
        + ArgminWeightedDot<O::Param, O::Float, O::Hessian>
        + ArgminNorm<F>
        + ArgminDot<O::Param, O::Float>
        + ArgminAdd<O::Param, O::Param>
        + ArgminSub<O::Param, O::Param>,
    O::Hessian: ArgminInv<O::Hessian> + ArgminDot<O::Param, O::Param>,
    F: ArgminFloat,
{
    const NAME: &'static str = "Dogleg";

    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &IterState<O>,
    ) -> Result<ArgminIterData<O>, Error> {
        let param = state.get_param();
        let g = state
            .get_grad()
            .unwrap_or_else(|| op.gradient(&param).unwrap());
        let h = state
            .get_hessian()
            .unwrap_or_else(|| op.hessian(&param).unwrap());
        let pstar;

        // pb = -H^-1g
        let pb = (h.inv()?).dot(&g).mul(&F::from_f64(-1.0).unwrap());

        if pb.norm() <= self.radius {
            pstar = pb;
        } else {
            // pu = - (g^Tg)/(g^THg) * g
            let pu = g.mul(&(-g.dot(&g) / g.weighted_dot(&h, &g)));
            // println!("pb: {:?}, pu: {:?}", pb, pu);

            let utu = pu.dot(&pu);
            let btb = pb.dot(&pb);
            let utb = pu.dot(&pb);

            // compute tau
            let delta = self.radius.powi(2);
            let t1 = F::from_f64(3.0).unwrap() * utb - btb - F::from_f64(2.0).unwrap() * utu;
            let t2 = (utb.powi(2) - F::from_f64(2.0).unwrap() * utb * delta + delta * btb
                - btb * utu
                + delta * utu)
                .sqrt();
            let t3 = F::from_f64(-2.0).unwrap() * utb + btb + utu;
            let tau1: F = -(t1 + t2) / t3;
            let tau2: F = -(t1 - t2) / t3;

            // pick maximum value of both -- not sure if this is the proper way
            let mut tau = tau1.max(tau2);

            // if calculation failed because t3 is too small, use the third option
            if tau.is_nan() || tau.is_infinite() {
                tau = (delta + btb - F::from_f64(2.0).unwrap() * utu) / (btb - utu);
            }

            if tau >= F::from_f64(0.0).unwrap() && tau < F::from_f64(1.0).unwrap() {
                pstar = pu.mul(&tau);
            } else if tau >= F::from_f64(1.0).unwrap() && tau <= F::from_f64(2.0).unwrap() {
                pstar = pu.add(&pb.sub(&pu).mul(&(tau - F::from_f64(1.0).unwrap())));
            } else {
                return Err(ArgminError::ImpossibleError {
                    text: "tau is bigger than 2, this is not supposed to happen.".to_string(),
                }
                .into());
            }
        }
        let out = ArgminIterData::new().param(pstar);
        Ok(out)
    }

    fn terminate(&mut self, state: &IterState<O>) -> TerminationReason {
        if state.get_iter() >= 1 {
            TerminationReason::MaxItersReached
        } else {
            TerminationReason::NotTerminated
        }
    }
}

impl<F: ArgminFloat> ArgminTrustRegion<F> for Dogleg<F> {
    fn set_radius(&mut self, radius: F) {
        self.radius = radius;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_trait_impl;

    test_trait_impl!(dogleg, Dogleg<f64>);
}
