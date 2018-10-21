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
    /// initial residual
    r_0: T,
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
            epsilon: 10e-10,
            p: T::default(),
            r: T::default(),
            r_0: T::default(),
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
        self.cur_grad().dot(p.clone()) + 0.5 * p.weighted_dot(self.cur_hessian(), p.clone())
    }

    /// calculate all possible step lengths
    fn tau<F>(&self, filter: F) -> f64
    where
        F: Fn(f64) -> bool,
    {
        let a = self.p.dot(self.p.clone());
        let b = self.d.dot(self.d.clone());
        let c = self.p.dot(self.d.clone());
        let delta = self.radius.powi(2);
        let t1 = (-a * b + b * delta + c.powi(2)).sqrt();
        let tau1 = -(t1 + c) / b;
        let tau2 = (t1 - c) / b;
        let mut t = vec![tau1, tau2];
        // Maybe calculating tau3 should only be done if b is close to zero?
        if b.abs() < 10e-8 || tau1.is_nan() || tau2.is_nan() {
            let tau3 = (delta - a) / (2.0 * c);
            t.push(tau3);
        }
        // remove NAN taus and calculate m (without f_init) for all taus, then sort them based on
        // their result and return the tau which corresponds to the lowest m
        let mut v = t
            .iter()
            .cloned()
            .enumerate()
            .filter(|(_, tau)| !tau.is_nan() && filter(*tau))
            .map(|(i, tau)| {
                let p = self.p.add(self.d.scale(tau));
                (i, self.eval_m(p))
            })
            .filter(|(_, m)| !m.is_nan())
            .collect::<Vec<(usize, f64)>>();
        v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
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
        self.base_reset();

        self.r_0 = self.cur_grad();
        self.d = self.r_0.scale(-1.0);
        self.p = self.r_0.zero();
        self.r = self.r_0.clone();

        if self.r_0.norm() < self.epsilon {
            self.set_termination_reason(TerminationReason::TargetPrecisionReached);
            let best = self.p.clone();
            self.set_best_param(best);
        }

        Ok(())
    }

    fn next_iter(&mut self) -> Result<ArgminIterationData<Self::Parameters>, Error> {
        let h = self.cur_hessian();

        // Current search direction d is a direction of zero curvature or negative curvature
        if self.d.weighted_dot(h.clone(), self.d.clone()) <= 0.0 {
            let tau = self.tau(|_| true);
            self.set_termination_reason(TerminationReason::TargetPrecisionReached);
            return Ok(ArgminIterationData::new(
                self.p.add(self.d.scale(tau)),
                std::f64::NEG_INFINITY,
            ));
        }

        let alpha = self.r.dot(self.r.clone()) / self.d.weighted_dot(h.clone(), self.d.clone());
        let p_n = self.p.add(self.d.scale(alpha));

        // new p violates trust region bound
        if p_n.norm() >= self.radius {
            let tau = self.tau(|x| x >= 0.0);
            self.set_termination_reason(TerminationReason::TargetPrecisionReached);
            return Ok(ArgminIterationData::new(self.p.add(self.d.scale(tau)), 0.0));
        }

        let r_n = self.r.add(h.dot(self.d.clone()).scale(alpha));
        if r_n.norm() < self.epsilon * self.r_0.norm() {
            self.set_termination_reason(TerminationReason::TargetPrecisionReached);
            return Ok(ArgminIterationData::new(p_n, std::f64::NEG_INFINITY));
        }

        let beta = r_n.dot(r_n.clone()) / self.r.dot(self.r.clone());
        self.d = r_n.add(self.d.scale(beta));
        self.r = r_n;
        self.p = p_n;

        Ok(ArgminIterationData::new(
            self.p.clone(),
            std::f64::NEG_INFINITY,
        ))
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
