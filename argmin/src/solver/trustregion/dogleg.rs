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
    ArgminError, ArgminFloat, ArgminKV, ArgminTrustRegion, Error, Gradient, Hessian, IterState,
    OpWrapper, Solver, State, TerminationReason,
};
use argmin_math::{
    ArgminAdd, ArgminDot, ArgminInv, ArgminMul, ArgminNorm, ArgminSub, ArgminWeightedDot,
};
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

impl<F> Dogleg<F>
where
    F: ArgminFloat,
{
    /// Constructor
    pub fn new() -> Self {
        Dogleg { radius: F::nan() }
    }
}

impl<O, F, P, H> Solver<O, IterState<P, P, (), H, F>> for Dogleg<F>
where
    O: Gradient<Param = P, Gradient = P> + Hessian<Param = P, Hessian = H>,
    P: Clone
        + ArgminMul<F, P>
        + ArgminNorm<F>
        + ArgminDot<P, F>
        + ArgminAdd<P, P>
        + ArgminSub<P, P>,
    H: ArgminInv<H> + ArgminDot<P, P>,
    F: ArgminFloat,
{
    const NAME: &'static str = "Dogleg";

    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        mut state: IterState<P, P, (), H, F>,
    ) -> Result<(IterState<P, P, (), H, F>, Option<ArgminKV>), Error> {
        let param = state.take_param().unwrap();
        let g = state
            .take_grad()
            .map(Result::Ok)
            .unwrap_or_else(|| op.gradient(&param))?;
        let h = state
            .take_hessian()
            .map(Result::Ok)
            .unwrap_or_else(|| op.hessian(&param))?;
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
        Ok((state.param(pstar).grad(g).hessian(h), None))
    }

    fn terminate(&mut self, state: &IterState<P, P, (), H, F>) -> TerminationReason {
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
