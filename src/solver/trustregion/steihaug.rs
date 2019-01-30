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

/// The Steihaug method is a conjugate gradients based approach for finding an approximate solution
/// to the second order approximation of the cost function within the trust region.
///
/// # References:
///
/// [0] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(ArgminSolver)]
pub struct Steihaug<'a, T, H>
where
    T: Clone
        + std::default::Default
        + std::fmt::Debug
        + ArgminWeightedDot<T, f64, H>
        + ArgminDot<T, f64>
        + ArgminAdd<T, T>
        + ArgminSub<T, T>
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
    /// r^Tr
    rtr: f64,
    /// initial residual
    r_0_norm: f64,
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
        + ArgminAdd<T, T>
        + ArgminSub<T, T>
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
        operator: &'a ArgminOperator<Parameters = T, OperatorOutput = f64, Hessian = H>,
    ) -> Self {
        let base = ArgminBase::new(operator, T::default());
        Steihaug {
            radius: std::f64::NAN,
            epsilon: 10e-10,
            p: T::default(),
            r: T::default(),
            rtr: std::f64::NAN,
            r_0_norm: std::f64::NAN,
            d: T::default(),
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
    fn eval_m(&self, p: &T) -> f64 {
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
                    let p = self.p.add(&self.d.scale(tau));
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

impl<'a, T, H> ArgminNextIter for Steihaug<'a, T, H>
where
    T: Clone
        + std::default::Default
        + std::fmt::Debug
        + ArgminWeightedDot<T, f64, H>
        + ArgminDot<T, f64>
        + ArgminAdd<T, T>
        + ArgminSub<T, T>
        + ArgminNorm<f64>
        + ArgminZero
        + ArgminScale<f64>,
    H: Clone + std::default::Default + ArgminDot<T, T>,
{
    type Parameters = T;
    type OperatorOutput = f64;
    type Hessian = H;

    fn init(&mut self) -> Result<(), Error> {
        self.base_reset();

        self.r = self.cur_grad();
        self.r_0_norm = self.r.norm();
        self.rtr = self.r.dot(&self.r);
        self.d = self.r.scale(-1.0);
        self.p = self.r.zero_like();

        if self.r_0_norm < self.epsilon {
            self.set_termination_reason(TerminationReason::TargetPrecisionReached);
            let best = self.p.clone();
            self.set_best_param(best);
        }

        Ok(())
    }

    fn next_iter(&mut self) -> Result<ArgminIterationData<Self::Parameters>, Error> {
        let h = self.cur_hessian();
        let dhd = self.d.weighted_dot(&h, &self.d);

        // Current search direction d is a direction of zero curvature or negative curvature
        if dhd <= 0.0 {
            let tau = self.tau(|_| true, true);
            self.set_termination_reason(TerminationReason::TargetPrecisionReached);
            return Ok(ArgminIterationData::new(
                self.p.add(&self.d.scale(tau)),
                0.0,
            ));
        }

        let alpha = self.rtr / dhd;
        let p_n = self.p.add(&self.d.scale(alpha));

        // new p violates trust region bound
        if p_n.norm() >= self.radius {
            let tau = self.tau(|x| x >= 0.0, false);
            self.set_termination_reason(TerminationReason::TargetPrecisionReached);
            return Ok(ArgminIterationData::new(
                self.p.add(&self.d.scale(tau)),
                0.0,
            ));
        }

        let r_n = self.r.add(&h.dot(&self.d).scale(alpha));

        if r_n.norm() < self.epsilon * self.r_0_norm {
            self.set_termination_reason(TerminationReason::TargetPrecisionReached);
            return Ok(ArgminIterationData::new(p_n, 0.0));
        }

        let rjtrj = r_n.dot(&r_n);
        let beta = rjtrj / self.rtr;
        self.d = r_n.add(&self.d.scale(beta));
        self.r = r_n;
        self.p = p_n;
        self.rtr = rjtrj;

        Ok(ArgminIterationData::new(self.p.clone(), 0.0))
    }
}

impl<'a, T, H> ArgminTrustRegion for Steihaug<'a, T, H>
where
    T: Clone
        + std::default::Default
        + std::fmt::Debug
        + ArgminWeightedDot<T, f64, H>
        + ArgminDot<T, f64>
        + ArgminAdd<T, T>
        + ArgminSub<T, T>
        + ArgminNorm<f64>
        + ArgminZero
        + ArgminScale<f64>,
    H: Clone + std::default::Default + ArgminDot<T, T>,
{
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
