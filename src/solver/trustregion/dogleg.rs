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

/// The Dogleg method computes the intersection of the trust region boundary with a path given by
/// the unconstraind minimum along the steepest descent direction and the optimum of the quadratic
/// approximation of the cost function at the current point.
///
/// # References:
///
/// [0] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(ArgminSolver, Serialize, Deserialize)]
pub struct Dogleg<O>
where
    O: ArgminOp<Output = f64>,
    O::Param: ArgminMul<f64, O::Param>
        + ArgminWeightedDot<O::Param, f64, O::Hessian>
        + ArgminNorm<f64>
        + ArgminDot<O::Param, f64>
        + ArgminAdd<O::Param, O::Param>
        + ArgminSub<O::Param, O::Param>,
    O::Hessian: ArgminInv<O::Hessian> + ArgminDot<O::Param, O::Param>,
{
    /// Radius
    radius: f64,
    /// base
    base: ArgminBase<O>,
}

impl<O> Dogleg<O>
where
    O: ArgminOp<Output = f64>,
    O::Param: ArgminMul<f64, O::Param>
        + ArgminWeightedDot<O::Param, f64, O::Hessian>
        + ArgminNorm<f64>
        + ArgminDot<O::Param, f64>
        + ArgminAdd<O::Param, O::Param>
        + ArgminSub<O::Param, O::Param>,
    O::Hessian: ArgminInv<O::Hessian> + ArgminDot<O::Param, O::Param>,
{
    /// Constructor
    ///
    /// Parameters:
    ///
    /// `operator`: operator
    pub fn new(operator: O) -> Self {
        let base = ArgminBase::new(operator, O::Param::default());
        Dogleg {
            radius: std::f64::NAN,
            base,
        }
    }
}

impl<O> ArgminIter for Dogleg<O>
where
    O: ArgminOp<Output = f64>,
    O::Param: ArgminMul<f64, O::Param>
        + ArgminWeightedDot<O::Param, f64, O::Hessian>
        + ArgminNorm<f64>
        + ArgminDot<O::Param, f64>
        + ArgminAdd<O::Param, O::Param>
        + ArgminSub<O::Param, O::Param>,
    O::Hessian: ArgminInv<O::Hessian> + ArgminDot<O::Param, O::Param>,
{
    type Param = O::Param;
    type Output = O::Output;
    type Hessian = O::Hessian;

    fn init(&mut self) -> Result<(), Error> {
        self.base_reset();
        // This is not an iterative method.
        self.set_max_iters(1);
        Ok(())
    }

    fn next_iter(&mut self) -> Result<ArgminIterData<Self::Param>, Error> {
        let g = self.cur_grad();
        let h = self.cur_hessian();
        let pstar;

        // pb = -H^-1g
        let pb = (self.cur_hessian().inv()?)
            .dot(&self.cur_grad())
            .mul(&(-1.0));

        if pb.norm() <= self.radius {
            pstar = pb;
        } else {
            // pu = - (g^Tg)/(g^THg) * g
            let pu = g.mul(&(-g.dot(&g) / g.weighted_dot(&h, &g)));

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
                pstar = pu.mul(&tau);
            } else if tau >= 1.0 && tau <= 2.0 {
                // pstar = pu + (tau - 1.0) * (pb - pu)
                pstar = pu.add(&pb.sub(&pu).mul(&(tau - 1.0)));
            } else {
                return Err(ArgminError::ImpossibleError {
                    text: "tau is bigger than 2, this is not supposed to happen.".to_string(),
                }
                .into());
            }
        }
        let out = ArgminIterData::new(pstar, 0.0);
        Ok(out)
    }
}

impl<O> ArgminTrustRegion for Dogleg<O>
where
    O: ArgminOp<Output = f64>,
    O::Param: ArgminMul<f64, O::Param>
        + ArgminWeightedDot<O::Param, f64, O::Hessian>
        + ArgminNorm<f64>
        + ArgminDot<O::Param, f64>
        + ArgminAdd<O::Param, O::Param>
        + ArgminSub<O::Param, O::Param>,
    O::Hessian: ArgminInv<O::Hessian> + ArgminDot<O::Param, O::Param>,
{
    // fn set_initial_parameter(&mut self, param: T) {
    //     self.set_cur_param(param);
    // }

    fn set_radius(&mut self, radius: f64) {
        self.radius = radius;
    }

    fn set_grad(&mut self, grad: O::Param) {
        self.set_cur_grad(grad);
    }

    fn set_hessian(&mut self, hessian: O::Hessian) {
        self.set_cur_hessian(hessian);
    }
}
