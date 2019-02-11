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

/// The Steihaug method is a conjugate gradients based approach for finding an approximate solution
/// to the second order approximation of the cost function within the trust region.
///
/// # References:
///
/// [0] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(ArgminSolver, Serialize, Deserialize)]
pub struct Steihaug<O>
where
    O: ArgminOp<Output = f64>,
    <O as ArgminOp>::Param:
        ArgminMul<f64, <O as ArgminOp>::Param>
            + ArgminWeightedDot<<O as ArgminOp>::Param, f64, <O as ArgminOp>::Hessian>
            + ArgminNorm<f64>
            + ArgminDot<<O as ArgminOp>::Param, f64>
            + ArgminAdd<<O as ArgminOp>::Param, <O as ArgminOp>::Param>
            + ArgminSub<<O as ArgminOp>::Param, <O as ArgminOp>::Param>
            + ArgminZero
            + ArgminMul<f64, <O as ArgminOp>::Param>,
    <O as ArgminOp>::Hessian: ArgminDot<<O as ArgminOp>::Param, <O as ArgminOp>::Param>,
{
    /// Radius
    radius: f64,
    /// epsilon
    epsilon: f64,
    /// p
    p: <O as ArgminOp>::Param,
    /// residual
    r: <O as ArgminOp>::Param,
    /// r^Tr
    rtr: f64,
    /// initial residual
    r_0_norm: f64,
    /// direction
    d: <O as ArgminOp>::Param,
    /// base
    base: ArgminBase<O>,
}

impl<O> Steihaug<O>
where
    O: ArgminOp<Output = f64>,
    <O as ArgminOp>::Param:
        ArgminMul<f64, <O as ArgminOp>::Param>
            + ArgminWeightedDot<<O as ArgminOp>::Param, f64, <O as ArgminOp>::Hessian>
            + ArgminNorm<f64>
            + ArgminDot<<O as ArgminOp>::Param, f64>
            + ArgminAdd<<O as ArgminOp>::Param, <O as ArgminOp>::Param>
            + ArgminSub<<O as ArgminOp>::Param, <O as ArgminOp>::Param>
            + ArgminZero
            + ArgminMul<f64, <O as ArgminOp>::Param>,
    <O as ArgminOp>::Hessian: ArgminDot<<O as ArgminOp>::Param, <O as ArgminOp>::Param>,
{
    /// Constructor
    ///
    /// Parameters:
    ///
    /// `operator`: operator
    pub fn new(operator: O) -> Self {
        let base = ArgminBase::new(operator, <O as ArgminOp>::Param::default());
        Steihaug {
            radius: std::f64::NAN,
            epsilon: 10e-10,
            p: <O as ArgminOp>::Param::default(),
            r: <O as ArgminOp>::Param::default(),
            rtr: std::f64::NAN,
            r_0_norm: std::f64::NAN,
            d: <O as ArgminOp>::Param::default(),
            base,
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
    fn eval_m(&self, p: &<O as ArgminOp>::Param) -> f64 {
        self.cur_grad().dot(&p) + 0.5 * p.weighted_dot(&self.cur_hessian(), &p)
    }

    /// calculate all possible step lengths
    #[allow(clippy::many_single_char_names)]
    fn tau<F>(&self, filter_func: F, eval: bool) -> f64
    where
        F: Fn(f64) -> bool,
    {
        let a = self.p.dot(&self.p);
        let b = self.d.dot(&self.d);
        let c = self.p.dot(&self.d);
        let delta = self.radius.powi(2);
        let t1 = (-a * b + b * delta + c.powi(2)).sqrt();
        let tau1 = -(t1 + c) / b;
        let tau2 = (t1 - c) / b;
        let mut t = vec![tau1, tau2];
        // Maybe calculating tau3 should only be done if b is close to zero?
        if b.abs() < 2.0 * std::f64::EPSILON || tau1.is_nan() || tau2.is_nan() {
            let tau3 = (delta - a) / (2.0 * c);
            t.push(tau3);
        }
        let v = if eval {
            // remove NAN taus and calculate m (without f_init) for all taus, then sort them based
            // on their result and return the tau which corresponds to the lowest m
            let mut v = t
                .iter()
                .cloned()
                .enumerate()
                .filter(|(_, tau)| !tau.is_nan() && filter_func(*tau))
                .map(|(i, tau)| {
                    let p = self.p.add(&self.d.mul(&tau));
                    (i, self.eval_m(&p))
                })
                .filter(|(_, m)| !m.is_nan())
                .collect::<Vec<(usize, f64)>>();
            v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            v
        } else {
            let mut v = t
                .iter()
                .cloned()
                .enumerate()
                .filter(|(_, tau)| !tau.is_nan() && filter_func(*tau))
                .collect::<Vec<(usize, f64)>>();
            v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            v
        };

        t[v[0].0]
    }
}

impl<O> ArgminIter for Steihaug<O>
where
    O: ArgminOp<Output = f64>,
    <O as ArgminOp>::Param:
        ArgminMul<f64, <O as ArgminOp>::Param>
            + ArgminWeightedDot<<O as ArgminOp>::Param, f64, <O as ArgminOp>::Hessian>
            + ArgminNorm<f64>
            + ArgminDot<<O as ArgminOp>::Param, f64>
            + ArgminAdd<<O as ArgminOp>::Param, <O as ArgminOp>::Param>
            + ArgminSub<<O as ArgminOp>::Param, <O as ArgminOp>::Param>
            + ArgminZero
            + ArgminMul<f64, <O as ArgminOp>::Param>,
    <O as ArgminOp>::Hessian: ArgminDot<<O as ArgminOp>::Param, <O as ArgminOp>::Param>,
{
    type Param = <O as ArgminOp>::Param;
    type Output = <O as ArgminOp>::Output;
    type Hessian = <O as ArgminOp>::Hessian;

    fn init(&mut self) -> Result<(), Error> {
        self.base_reset();

        self.r = self.cur_grad();
        self.r_0_norm = self.r.norm();
        self.rtr = self.r.dot(&self.r);
        self.d = self.r.mul(&(-1.0));
        self.p = self.r.zero_like();

        if self.r_0_norm < self.epsilon {
            self.set_termination_reason(TerminationReason::TargetPrecisionReached);
            let best = self.p.clone();
            self.set_best_param(best);
        }

        Ok(())
    }

    fn next_iter(&mut self) -> Result<ArgminIterData<Self::Param>, Error> {
        let h = self.cur_hessian();
        let dhd = self.d.weighted_dot(&h, &self.d);

        // Current search direction d is a direction of zero curvature or negative curvature
        if dhd <= 0.0 {
            let tau = self.tau(|_| true, true);
            self.set_termination_reason(TerminationReason::TargetPrecisionReached);
            return Ok(ArgminIterData::new(self.p.add(&self.d.mul(&tau)), 0.0));
        }

        let alpha = self.rtr / dhd;
        let p_n = self.p.add(&self.d.mul(&alpha));

        // new p violates trust region bound
        if p_n.norm() >= self.radius {
            let tau = self.tau(|x| x >= 0.0, false);
            self.set_termination_reason(TerminationReason::TargetPrecisionReached);
            return Ok(ArgminIterData::new(self.p.add(&self.d.mul(&tau)), 0.0));
        }

        let r_n = self.r.add(&h.dot(&self.d).mul(&alpha));

        if r_n.norm() < self.epsilon * self.r_0_norm {
            self.set_termination_reason(TerminationReason::TargetPrecisionReached);
            return Ok(ArgminIterData::new(p_n, 0.0));
        }

        let rjtrj = r_n.dot(&r_n);
        let beta = rjtrj / self.rtr;
        self.d = r_n.add(&self.d.mul(&beta));
        self.r = r_n;
        self.p = p_n;
        self.rtr = rjtrj;

        Ok(ArgminIterData::new(self.p.clone(), 0.0))
    }
}

impl<O> ArgminTrustRegion for Steihaug<O>
where
    O: ArgminOp<Output = f64>,
    <O as ArgminOp>::Param:
        ArgminMul<f64, <O as ArgminOp>::Param>
            + ArgminWeightedDot<<O as ArgminOp>::Param, f64, <O as ArgminOp>::Hessian>
            + ArgminNorm<f64>
            + ArgminDot<<O as ArgminOp>::Param, f64>
            + ArgminAdd<<O as ArgminOp>::Param, <O as ArgminOp>::Param>
            + ArgminSub<<O as ArgminOp>::Param, <O as ArgminOp>::Param>
            + ArgminZero
            + ArgminMul<f64, <O as ArgminOp>::Param>,
    <O as ArgminOp>::Hessian: ArgminDot<<O as ArgminOp>::Param, <O as ArgminOp>::Param>,
{
    fn set_radius(&mut self, radius: f64) {
        self.radius = radius;
    }

    fn set_grad(&mut self, grad: <O as ArgminOp>::Param) {
        self.set_cur_grad(grad);
    }

    fn set_hessian(&mut self, hessian: <O as ArgminOp>::Hessian) {
        self.set_cur_hessian(hessian);
    }
}
