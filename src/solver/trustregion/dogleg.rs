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
use std;

/// The Dogleg method computes the intersection of the trust region boundary with a path given by
/// the unconstraind minimum along the steepest descent direction and the optimum of the quadratic
/// approximation of the cost function at the current point.
///
/// # References:
///
/// [0] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(ArgminSolver)]
pub struct Dogleg<'a, T, H>
where
    T: Clone
        + std::default::Default
        + std::fmt::Debug
        + ArgminWeightedDot<T, f64, H>
        + ArgminDot<T, f64>
        + ArgminAdd<T>
        + ArgminSub<T>
        + ArgminNorm<f64>
        + ArgminScale<f64>,
    H: Clone + std::default::Default + ArgminInv<H> + ArgminDot<T, T>,
{
    /// Radius
    radius: f64,
    /// base
    base: ArgminBase<'a, T, f64, H>,
}

impl<'a, T, H> Dogleg<'a, T, H>
where
    T: Clone
        + std::default::Default
        + std::fmt::Debug
        + ArgminWeightedDot<T, f64, H>
        + ArgminDot<T, f64>
        + ArgminAdd<T>
        + ArgminSub<T>
        + ArgminNorm<f64>
        + ArgminScale<f64>,
    H: Clone + std::default::Default + ArgminInv<H> + ArgminDot<T, T>,
{
    /// Constructor
    ///
    /// Parameters:
    ///
    /// `operator`: operator
    pub fn new(
        operator: &'a ArgminOperator<Parameters = T, OperatorOutput = f64, Hessian = H>,
    ) -> Self {
        let base = ArgminBase::new(operator, T::default());
        Dogleg {
            radius: std::f64::NAN,
            base,
        }
    }
}

impl<'a, T, H> ArgminNextIter for Dogleg<'a, T, H>
where
    T: Clone
        + std::default::Default
        + std::fmt::Debug
        + ArgminWeightedDot<T, f64, H>
        + ArgminDot<T, f64>
        + ArgminAdd<T>
        + ArgminSub<T>
        + ArgminNorm<f64>
        + ArgminScale<f64>,
    H: Clone + std::default::Default + ArgminInv<H> + ArgminDot<T, T>,
{
    type Parameters = T;
    type OperatorOutput = f64;
    type Hessian = H;

    fn init(&mut self) -> Result<(), Error> {
        self.base_reset();
        // This is not an iterative method.
        self.set_max_iters(1);
        Ok(())
    }

    fn next_iter(&mut self) -> Result<ArgminIterationData<Self::Parameters>, Error> {
        let g = self.cur_grad();
        let h = self.cur_hessian();
        let pstar;

        // pb = -H^-1g
        let pb = (self.cur_hessian().ainv()?)
            .dot(&self.cur_grad())
            .scale(-1.0);

        if pb.norm() <= self.radius {
            pstar = pb;
        } else {
            // pu = - (g^Tg)/(g^THg) * g
            let pu = g.scale(-g.dot(&g) / g.weighted_dot(&h, &g));

            let utu = pu.dot(&pu);
            let btb = pb.dot(&pb);
            let utb = pu.dot(&pb);

            // compute tau
            let delta = self.radius.powi(2);
            let t1 = 3.0 * utb - btb - 2.0 * utu;
            let t2 =
                (utb.powi(2) - 2.0 * utb * delta + delta * btb - btb * utu + delta * utu).sqrt();
            let t3 = -2.0 * utb + btb + utu;
            let tau1: f64 = -(t1 + t2) / t3;
            let tau2: f64 = -(t1 - t2) / t3;

            // pick maximum value of both -- not sure if this is the proper way
            let mut tau = tau1.max(tau2);

            // if calculation failed because t3 is too small, use the third option
            if tau.is_nan() {
                tau = (delta + btb - 2.0 * utu) / (btb - utu);
            }

            if tau >= 0.0 && tau < 1.0 {
                pstar = pu.scale(tau);
            } else if tau >= 1.0 && tau <= 2.0 {
                // pstar = pu + (tau - 1.0) * (pb - pu)
                pstar = pu.add(&pb.sub(&pu).scale(tau - 1.0));
            } else {
                return Err(ArgminError::ImpossibleError {
                    text: "tau is bigger than 2, this is not supposed to happen.".to_string(),
                }
                .into());
            }
        }
        let out = ArgminIterationData::new(pstar, 0.0);
        Ok(out)
    }
}

impl<'a, T, H> ArgminTrustRegion for Dogleg<'a, T, H>
where
    T: Clone
        + std::default::Default
        + std::fmt::Debug
        + ArgminWeightedDot<T, f64, H>
        + ArgminDot<T, f64>
        + ArgminAdd<T>
        + ArgminSub<T>
        + ArgminNorm<f64>
        + ArgminScale<f64>,
    H: Clone + std::default::Default + ArgminInv<H> + ArgminDot<T, T>,
{
    // fn set_initial_parameter(&mut self, param: T) {
    //     self.set_cur_param(param);
    // }

    fn set_radius(&mut self, radius: f64) {
        self.radius = radius;
    }

    fn set_grad(&mut self, grad: T) {
        self.set_cur_grad(grad);
    }

    fn set_hessian(&mut self, hessian: H) {
        self.set_cur_hessian(hessian);
    }
}
