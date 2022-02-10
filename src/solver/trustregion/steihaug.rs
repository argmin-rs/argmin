// Copyright 2018-2020 argmin developers
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
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

/// The Steihaug method is a conjugate gradients based approach for finding an approximate solution
/// to the second order approximation of the cost function within the trust region.
///
/// # References:
///
/// \[0\] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(Clone, Serialize, Deserialize, Debug, Copy, PartialEq, PartialOrd, Default)]
pub struct Steihaug<P, F> {
    /// Radius
    radius: F,
    /// epsilon
    epsilon: F,
    /// p
    p: Option<P>,
    /// residual
    r: Option<P>,
    /// r^Tr
    rtr: F,
    /// initial residual
    r_0_norm: F,
    /// direction
    d: Option<P>,
    /// max iters
    max_iters: u64,
}

impl<P, F> Steihaug<P, F>
where
    P: Clone + ArgminMul<F, P> + ArgminDot<P, F> + ArgminAdd<P, P>,
    F: ArgminFloat,
{
    /// Constructor
    pub fn new() -> Self {
        Steihaug {
            radius: F::nan(),
            epsilon: F::from_f64(10e-10).unwrap(),
            p: None,
            r: None,
            rtr: F::nan(),
            r_0_norm: F::nan(),
            d: None,
            max_iters: std::u64::MAX,
        }
    }

    /// Set epsilon
    pub fn epsilon(mut self, epsilon: F) -> Result<Self, Error> {
        if epsilon <= F::from_f64(0.0).unwrap() {
            return Err(ArgminError::InvalidParameter {
                text: "Steihaug: epsilon must be > 0.0.".to_string(),
            }
            .into());
        }
        self.epsilon = epsilon;
        Ok(self)
    }

    /// set maximum number of iterations
    pub fn max_iters(mut self, iters: u64) -> Self {
        self.max_iters = iters;
        self
    }

    /// evaluate m(p) (without considering f_init because it is not available)
    fn eval_m<H>(&self, p: &P, g: &P, h: &H) -> F
    where
        P: ArgminWeightedDot<P, F, H>,
    {
        // self.cur_grad().dot(&p) + 0.5 * p.weighted_dot(&self.cur_hessian(), &p)
        g.dot(p) + F::from_f64(0.5).unwrap() * p.weighted_dot(h, p)
    }

    /// calculate all possible step lengths
    #[allow(clippy::many_single_char_names)]
    fn tau<G, H>(&self, filter_func: G, eval: bool, g: &P, h: &H) -> F
    where
        G: Fn(F) -> bool,
        H: ArgminDot<P, P>,
    {
        let p = self.p.as_ref().unwrap();
        let d = self.d.as_ref().unwrap();
        let a = p.dot(p);
        let b = d.dot(d);
        let c = p.dot(d);
        let delta = self.radius.powi(2);
        let t1 = (-a * b + b * delta + c.powi(2)).sqrt();
        let tau1 = -(t1 + c) / b;
        let tau2 = (t1 - c) / b;
        let mut t = vec![tau1, tau2];
        // Maybe calculating tau3 should only be done if b is close to zero?
        if tau1.is_nan() || tau2.is_nan() || tau1.is_infinite() || tau2.is_infinite() {
            let tau3 = (delta - a) / (F::from_f64(2.0).unwrap() * c);
            t.push(tau3);
        }
        let v = if eval {
            // remove NAN taus and calculate m (without f_init) for all taus, then sort them based
            // on their result and return the tau which corresponds to the lowest m
            let mut v = t
                .iter()
                .cloned()
                .enumerate()
                .filter(|(_, tau)| (!tau.is_nan() || !tau.is_infinite()) && filter_func(*tau))
                .map(|(i, tau)| {
                    let p_local = p.add(&d.mul(&tau));
                    (i, self.eval_m(&p_local, g, h))
                })
                .filter(|(_, m)| !m.is_nan() || !m.is_infinite())
                .collect::<Vec<(usize, F)>>();
            v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            v
        } else {
            let mut v = t
                .iter()
                .cloned()
                .enumerate()
                .filter(|(_, tau)| (!tau.is_nan() || !tau.is_infinite()) && filter_func(*tau))
                .collect::<Vec<(usize, F)>>();
            v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            v
        };

        t[v[0].0]
    }
}

impl<P, O, F> Solver<O> for Steihaug<P, F>
where
    O: ArgminOp<Param = P, Output = F, Float = F>,
    P: Clone
        + Serialize
        + DeserializeOwned
        + Default
        + ArgminMul<F, P>
        + ArgminWeightedDot<P, F, O::Hessian>
        + ArgminNorm<F>
        + ArgminDot<P, F>
        + ArgminAdd<P, P>
        + ArgminSub<P, P>
        + ArgminZeroLike
        + ArgminMul<F, P>,
    O::Hessian: ArgminDot<P, P>,
    F: ArgminFloat,
{
    const NAME: &'static str = "Steihaug";

    fn init(
        &mut self,
        _op: &mut OpWrapper<O>,
        state: &IterState<O>,
    ) -> Result<Option<ArgminIterData<O>>, Error> {
        let r = state.get_grad().unwrap();
        self.r = Some(r.clone());

        self.r_0_norm = r.norm();
        self.rtr = r.dot(&r);
        self.d = Some(r.mul(&F::from_f64(-1.0).unwrap()));
        let p = r.zero_like();
        self.p = Some(p.clone());

        Ok(if self.r_0_norm < self.epsilon {
            Some(
                ArgminIterData::new()
                    .param(p)
                    .termination_reason(TerminationReason::TargetPrecisionReached),
            )
        } else {
            None
        })
    }

    fn next_iter(
        &mut self,
        _op: &mut OpWrapper<O>,
        state: &IterState<O>,
    ) -> Result<ArgminIterData<O>, Error> {
        let grad = state.get_grad().unwrap();
        let h = state.get_hessian().unwrap();
        let d = self.d.as_ref().unwrap();
        let dhd = d.weighted_dot(&h, d);

        // Current search direction d is a direction of zero curvature or negative curvature
        let p = self.p.as_ref().unwrap();
        if dhd <= F::from_f64(0.0).unwrap() {
            let tau = self.tau(|_| true, true, &grad, &h);
            return Ok(ArgminIterData::new()
                .param(p.add(&d.mul(&tau)))
                .termination_reason(TerminationReason::TargetPrecisionReached));
        }

        let alpha = self.rtr / dhd;
        let p_n = p.add(&d.mul(&alpha));

        // new p violates trust region bound
        if p_n.norm() >= self.radius {
            let tau = self.tau(|x| x >= F::from_f64(0.0).unwrap(), false, &grad, &h);
            return Ok(ArgminIterData::new()
                .param(p.add(&d.mul(&tau)))
                .termination_reason(TerminationReason::TargetPrecisionReached));
        }

        let r = self.r.as_ref().unwrap();
        let r_n = r.add(&h.dot(d).mul(&alpha));

        if r_n.norm() < self.epsilon * self.r_0_norm {
            return Ok(ArgminIterData::new()
                .param(p_n)
                .termination_reason(TerminationReason::TargetPrecisionReached));
        }

        let rjtrj = r_n.dot(&r_n);
        let beta = rjtrj / self.rtr;
        self.d = Some(r_n.mul(&F::from_f64(-1.0).unwrap()).add(&d.mul(&beta)));
        self.r = Some(r_n);
        self.p = Some(p_n.clone());
        self.rtr = rjtrj;

        Ok(ArgminIterData::new()
            .param(p_n)
            .cost(self.rtr)
            .grad(grad)
            .hessian(h))
    }

    fn terminate(&mut self, state: &IterState<O>) -> TerminationReason {
        if state.get_iter() >= self.max_iters {
            TerminationReason::MaxItersReached
        } else {
            TerminationReason::NotTerminated
        }
    }
}

impl<P: Clone + Serialize, F: ArgminFloat> ArgminTrustRegion<F> for Steihaug<P, F> {
    fn set_radius(&mut self, radius: F) {
        self.radius = radius;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_trait_impl;

    test_trait_impl!(steihaug, Steihaug<MinimalNoOperator, f64>);
}
