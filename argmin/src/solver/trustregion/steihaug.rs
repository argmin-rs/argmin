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
    ArgminError, ArgminFloat, ArgminIterData, ArgminOp, ArgminTrustRegion, DeserializeOwnedAlias,
    Error, IterState, OpWrapper, SerializeAlias, Solver, TerminationReason,
};
use argmin_math::{
    ArgminAdd, ArgminDot, ArgminMul, ArgminNorm, ArgminSub, ArgminWeightedDot, ArgminZeroLike,
};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// The Steihaug method is a conjugate gradients based approach for finding an approximate solution
/// to the second order approximation of the cost function within the trust region.
///
/// # References:
///
/// \[0\] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(Clone, Default)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct Steihaug<P, F> {
    /// Radius
    radius: F,
    /// epsilon
    epsilon: F,
    /// p
    p: P,
    /// residual
    r: P,
    /// r^Tr
    rtr: F,
    /// initial residual
    r_0_norm: F,
    /// direction
    d: P,
    /// max iters
    max_iters: u64,
}

impl<P, F> Steihaug<P, F>
where
    P: Default + Clone + ArgminMul<F, P> + ArgminDot<P, F> + ArgminAdd<P, P>,
    F: ArgminFloat,
{
    /// Constructor
    pub fn new() -> Self {
        Steihaug {
            radius: F::nan(),
            epsilon: F::from_f64(10e-10).unwrap(),
            p: P::default(),
            r: P::default(),
            rtr: F::nan(),
            r_0_norm: F::nan(),
            d: P::default(),
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
    #[must_use]
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
        P: ArgminWeightedDot<P, F, H>,
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
                    let p = self.p.add(&self.d.mul(&tau));
                    (i, self.eval_m(&p, g, h))
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
        + SerializeAlias
        + DeserializeOwnedAlias
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
        self.r = state.get_grad().unwrap();

        self.r_0_norm = self.r.norm();
        self.rtr = self.r.dot(&self.r);
        self.d = self.r.mul(&F::from_f64(-1.0).unwrap());
        self.p = self.r.zero_like();

        Ok(if self.r_0_norm < self.epsilon {
            Some(
                ArgminIterData::new()
                    .param(self.p.clone())
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
        let dhd = self.d.weighted_dot(&h, &self.d);

        // Current search direction d is a direction of zero curvature or negative curvature
        if dhd <= F::from_f64(0.0).unwrap() {
            let tau = self.tau(|_| true, true, &grad, &h);
            return Ok(ArgminIterData::new()
                .param(self.p.add(&self.d.mul(&tau)))
                .termination_reason(TerminationReason::TargetPrecisionReached));
        }

        let alpha = self.rtr / dhd;
        let p_n = self.p.add(&self.d.mul(&alpha));

        // new p violates trust region bound
        if p_n.norm() >= self.radius {
            let tau = self.tau(|x| x >= F::from_f64(0.0).unwrap(), false, &grad, &h);
            return Ok(ArgminIterData::new()
                .param(self.p.add(&self.d.mul(&tau)))
                .termination_reason(TerminationReason::TargetPrecisionReached));
        }

        let r_n = self.r.add(&h.dot(&self.d).mul(&alpha));

        if r_n.norm() < self.epsilon * self.r_0_norm {
            return Ok(ArgminIterData::new()
                .param(p_n)
                .termination_reason(TerminationReason::TargetPrecisionReached));
        }

        let rjtrj = r_n.dot(&r_n);
        let beta = rjtrj / self.rtr;
        self.d = r_n.mul(&F::from_f64(-1.0).unwrap()).add(&self.d.mul(&beta));
        self.r = r_n;
        self.p = p_n;
        self.rtr = rjtrj;

        Ok(ArgminIterData::new()
            .param(self.p.clone())
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

impl<P: Clone + SerializeAlias, F: ArgminFloat> ArgminTrustRegion<F> for Steihaug<P, F> {
    fn set_radius(&mut self, radius: F) {
        self.radius = radius;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::MinimalNoOperator;
    use crate::test_trait_impl;

    test_trait_impl!(steihaug, Steihaug<MinimalNoOperator, f64>);
}
