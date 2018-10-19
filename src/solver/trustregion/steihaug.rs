// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Steihaug method
//!
//!
//! ## Reference
//!
//! TODO
//!
// //!
// //! # Example
// //!
// //! ```rust
// //! todo
// //! ```

use prelude::*;
use std;

/// Steihaug method
#[derive(ArgminSolver)]
pub struct Steihaug<'a, T, H>
where
    T: Clone
        + std::default::Default
        + std::fmt::Debug
        + ArgminWeightedDot<T, f64, H>
        + ArgminDot<T, f64>
        + ArgminAdd<T>
        + ArgminSub<T>
        + ArgminNorm<f64>
        + ArgminZero
        + ArgminScale<f64>,
    H: Clone + std::default::Default + ArgminDot<T, T>,
{
    /// Radius
    radius: f64,
    /// epsilon
    epsilon: f64,
    /// p
    p: T,
    /// residual
    r: T,
    /// direction
    d: T,
    /// base
    base: ArgminBase<'a, T, f64, H>,
}

impl<'a, T, H> Steihaug<'a, T, H>
where
    T: Clone
        + std::default::Default
        + std::fmt::Debug
        + ArgminWeightedDot<T, f64, H>
        + ArgminDot<T, f64>
        + ArgminAdd<T>
        + ArgminSub<T>
        + ArgminNorm<f64>
        + ArgminZero
        + ArgminScale<f64>,
    H: Clone + std::default::Default + ArgminDot<T, T>,
{
    /// Constructor
    ///
    /// Parameters:
    ///
    /// `operator`: operator
    pub fn new(
        operator: Box<ArgminOperator<Parameters = T, OperatorOutput = f64, Hessian = H> + 'a>,
    ) -> Self {
        let base = ArgminBase::new(operator, T::default());
        Steihaug {
            radius: std::f64::NAN,
            epsilon: 10e-8,
            p: T::default(),
            r: T::default(),
            d: T::default(),
            base: base,
        }
    }

    /// Set epsilon
    pub fn set_epsilon(&mut self, epsilon: f64) -> Result<&mut Self, Error> {
        if epsilon <= 0.0 {
            return Err(ArgminError::InvalidParameter {
                text: "Steihaug: epsilon must be > 0.0.".to_string(),
            }
            .into());
        }
        self.epsilon = epsilon;
        Ok(self)
    }

    /// evaluate m(p) (without considering f_init because it is not available)
    fn eval_m(&self, p: T) -> f64 {
        self.base.cur_grad().dot(p.clone())
            + 0.5 * p.weighted_dot(self.base.cur_hessian(), p.clone())
    }

    /// calculate all possible step lengths
    fn tau(&self) -> f64 {
        let a = self.p.dot(self.p.clone());
        let b = self.d.dot(self.d.clone());
        let c = self.p.dot(self.d.clone());
        let delta = self.radius.powi(2);
        let t1 = (-a * b + b * delta + c.powi(2)).sqrt();
        let tau1 = -(t1 + c) / b;
        let tau2 = (t1 - c) / b;
        let tau3 = (delta - a) / (2.0 * c);
        let t = vec![tau1, tau2, tau3];
        let mut v = t
            .iter()
            .cloned()
            .enumerate()
            .filter(|(_, tau)| !tau.is_nan())
            .map(|(i, tau)| {
                let p = self.p.add(self.d.scale(tau));
                (i, self.eval_m(p))
            })
            .collect::<Vec<(usize, f64)>>();
        v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        v[0].1
    }
}

impl<'a, T, H> ArgminNextIter for Steihaug<'a, T, H>
where
    T: Clone
        + std::default::Default
        + std::fmt::Debug
        + ArgminWeightedDot<T, f64, H>
        + ArgminDot<T, f64>
        + ArgminAdd<T>
        + ArgminSub<T>
        + ArgminNorm<f64>
        + ArgminZero
        + ArgminScale<f64>,
    H: Clone + std::default::Default + ArgminDot<T, T>,
{
    type Parameters = T;
    type OperatorOutput = f64;
    type Hessian = H;

    fn init(&mut self) -> Result<(), Error> {
        self.base.reset();

        self.r = self.base.cur_grad();
        self.d = self.r.scale(-1.0);
        self.p = self.r.zero();

        if self.p.norm() < self.epsilon {
            self.base
                .set_termination_reason(TerminationReason::TargetPrecisionReached);
            self.base.set_best_param(self.p.clone());
        }

        Ok(())
    }

    fn next_iter(&mut self) -> Result<ArgminIterationData<Self::Parameters>, Error> {
        // Current search direction d is a direction of zero curvature or negative curvature
        if self.d.weighted_dot(self.base.cur_hessian(), self.d.clone()) <= 0.0 {
            let tau = self.tau();
            self.base
                .set_termination_reason(TerminationReason::TargetPrecisionReached);
            return Ok(ArgminIterationData::new(self.p.add(self.d.scale(tau)), 0.0));
        }

        unimplemented!()
        // let g = self.base.cur_grad();
        // let h = self.base.cur_hessian();
        // let pstar;
        //
        // // pb = -H^-1g
        // let pb = (self.base.cur_hessian().ainv()?)
        //     .dot(self.base.cur_grad())
        //     .scale(-1.0);
        //
        // if pb.norm() <= self.radius {
        //     pstar = pb;
        // } else {
        //     // pu = - (g^Tg)/(g^THg) * g
        //     let pu = g.scale(-g.dot(g.clone()) / g.weighted_dot(h.clone(), g.clone()));
        //
        //     let utu = pu.dot(pu.clone());
        //     let btb = pb.dot(pb.clone());
        //     let utb = pu.dot(pb.clone());
        //
        //     // compute tau
        //     let delta = self.radius.powi(2);
        //     let t1 = 3.0 * utb - btb - 2.0 * utu;
        //     let t2 =
        //         (utb.powi(2) - 2.0 * utb * delta + delta * btb - btb * utu + delta * utu).sqrt();
        //     let t3 = -2.0 * utb + btb + utu;
        //     let tau1: f64 = -(t1 + t2) / t3;
        //     let tau2: f64 = -(t1 - t2) / t3;
        //
        //     // pick maximum value of both -- not sure if this is the proper way
        //     let mut tau = tau1.max(tau2);
        //
        //     // if calculation failed because t3 is too small, use the third option
        //     if tau.is_nan() {
        //         tau = (delta + btb - 2.0 * utu) / (btb - utu);
        //     }
        //
        //     if tau >= 0.0 && tau < 1.0 {
        //         pstar = pu.scale(tau);
        //     } else if tau >= 1.0 && tau <= 2.0 {
        //         // pstar = pu + (tau - 1.0) * (pb - pu)
        //         pstar = pu.add(pb.sub(pu.clone()).scale(tau - 1.0));
        //     } else {
        //         return Err(ArgminError::ImpossibleError {
        //             text: "tau is bigger than 2, this is not supposed to happen.".to_string(),
        //         }
        //         .into());
        //     }
        // }
        // let out = ArgminIterationData::new(pstar, 0.0);
        // Ok(out)
    }
}

impl<'a, T, H> ArgminTrustRegion for Steihaug<'a, T, H>
where
    T: Clone
        + std::default::Default
        + std::fmt::Debug
        + ArgminWeightedDot<T, f64, H>
        + ArgminDot<T, f64>
        + ArgminAdd<T>
        + ArgminSub<T>
        + ArgminNorm<f64>
        + ArgminZero
        + ArgminScale<f64>,
    H: Clone + std::default::Default + ArgminDot<T, T>,
{
    // fn set_initial_parameter(&mut self, param: T) {
    //     self.base.set_cur_param(param);
    // }

    fn set_radius(&mut self, radius: f64) {
        self.radius = radius;
    }

    fn set_grad(&mut self, grad: T) {
        self.base.set_cur_grad(grad);
    }

    fn set_hessian(&mut self, hessian: H) {
        self.base.set_cur_hessian(hessian);
    }
}
