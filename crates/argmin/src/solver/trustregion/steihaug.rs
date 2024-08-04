// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::{
    ArgminFloat, Error, IterState, Problem, Solver, State, TerminationReason, TerminationStatus,
    TrustRegionRadius, KV,
};
use argmin_math::{
    ArgminAdd, ArgminDot, ArgminL2Norm, ArgminMul, ArgminWeightedDot, ArgminZeroLike,
};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// # Steihaug method
///
/// The Steihaug method is a conjugate gradients based approach for finding an approximate solution
/// to the second order approximation of the cost function within the trust region.
///
/// ## Reference
///
/// Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(Clone, Default)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
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
    P: ArgminMul<F, P> + ArgminDot<P, F> + ArgminAdd<P, P>,
    F: ArgminFloat,
{
    /// Construct a new instance of [`Steihaug`]
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::trustregion::Steihaug;
    /// let sh: Steihaug<Vec<f64>, f64> = Steihaug::new();
    /// ```
    pub fn new() -> Self {
        Steihaug {
            radius: F::nan(),
            epsilon: float!(10e-10),
            p: None,
            r: None,
            rtr: F::nan(),
            r_0_norm: F::nan(),
            d: None,
            max_iters: u64::MAX,
        }
    }

    /// Set epsilon
    ///
    /// The algorithm stops when the residual is smaller than `epsilon`.
    ///
    /// Must be larger than 0 and defaults to 10^-10.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::trustregion::Steihaug;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// let sh: Steihaug<Vec<f64>, f64> = Steihaug::new().with_epsilon(10e-9)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_epsilon(mut self, epsilon: F) -> Result<Self, Error> {
        if epsilon <= float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`Steihaug`: epsilon must be > 0.0."
            ));
        }
        self.epsilon = epsilon;
        Ok(self)
    }

    /// Set maximum number of iterations
    ///
    /// The algorithm stops after `iter` iterations.
    ///
    /// Defaults to `u64::MAX`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::trustregion::Steihaug;
    /// # use argmin::core::Error;
    /// let sh: Steihaug<Vec<f64>, f64> = Steihaug::new().with_max_iters(100);
    /// ```
    #[must_use]
    pub fn with_max_iters(mut self, iters: u64) -> Self {
        self.max_iters = iters;
        self
    }

    /// evaluate m(p) (without considering f_init because it is not available)
    fn eval_m<H>(&self, p: &P, g: &P, h: &H) -> F
    where
        P: ArgminWeightedDot<P, F, H>,
    {
        g.dot(p) + float!(0.5) * p.weighted_dot(h, p)
    }

    /// calculate all possible step lengths
    #[allow(clippy::many_single_char_names)]
    fn tau<G, H>(&self, filter_func: G, eval: bool, g: &P, h: &H) -> F
    where
        G: Fn(F) -> bool,
        P: ArgminWeightedDot<P, F, H>,
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
            let tau3 = (delta - a) / (float!(2.0) * c);
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

impl<P, O, F, H> Solver<O, IterState<P, P, (), H, (), F>> for Steihaug<P, F>
where
    P: Clone
        + ArgminMul<F, P>
        + ArgminL2Norm<F>
        + ArgminDot<P, F>
        + ArgminAdd<P, P>
        + ArgminZeroLike,
    H: ArgminDot<P, P>,
    F: ArgminFloat,
{
    fn name(&self) -> &str {
        "Steihaug"
    }

    fn init(
        &mut self,
        _problem: &mut Problem<O>,
        state: IterState<P, P, (), H, (), F>,
    ) -> Result<(IterState<P, P, (), H, (), F>, Option<KV>), Error> {
        let r = state
            .get_gradient()
            .ok_or_else(argmin_error_closure!(
                NotInitialized,
                concat!(
                    "`Steihaug` requires an initial gradient. ",
                    "Please provide an initial gradient via `Executor`s `configure` method."
                )
            ))?
            .clone();

        if state.get_hessian().is_none() {
            return Err(argmin_error!(
                NotInitialized,
                concat!(
                    "`Steihaug` requires an initial Hessian. ",
                    "Please provide an initial Hessian via `Executor`s `configure` method."
                )
            ));
        }

        self.r_0_norm = r.l2_norm();
        self.rtr = r.dot(&r);
        self.d = Some(r.mul(&float!(-1.0)));
        let p = r.zero_like();
        self.p = Some(p.clone());

        self.r = Some(r);

        Ok((state.param(p), None))
    }

    fn next_iter(
        &mut self,
        _problem: &mut Problem<O>,
        mut state: IterState<P, P, (), H, (), F>,
    ) -> Result<(IterState<P, P, (), H, (), F>, Option<KV>), Error> {
        let grad = state.take_gradient().ok_or_else(argmin_error_closure!(
            PotentialBug,
            "`Steihaug`: Gradient in state not set."
        ))?;

        let h = state.take_hessian().ok_or_else(argmin_error_closure!(
            PotentialBug,
            "`Steihaug`: Hessian in state not set."
        ))?;

        let d = self.d.as_ref().unwrap();
        let dhd = d.weighted_dot(&h, d);

        // Current search direction d is a direction of zero curvature or negative curvature
        let p = self.p.as_ref().unwrap();
        if dhd <= float!(0.0) {
            let tau = self.tau(|_| true, true, &grad, &h);
            return Ok((
                state
                    .param(p.add(&d.mul(&tau)))
                    .terminate_with(TerminationReason::SolverConverged),
                None,
            ));
        }

        let alpha = self.rtr / dhd;
        let p_n = p.add(&d.mul(&alpha));

        // new p violates trust region bound
        if p_n.l2_norm() >= self.radius {
            let tau = self.tau(|x| x >= float!(0.0), false, &grad, &h);
            return Ok((
                state
                    .param(p.add(&d.mul(&tau)))
                    .terminate_with(TerminationReason::SolverConverged),
                None,
            ));
        }

        let r = self.r.as_ref().unwrap();
        let r_n = r.add(&h.dot(d).mul(&alpha));

        if r_n.l2_norm() < self.epsilon * self.r_0_norm {
            return Ok((
                state
                    .param(p_n)
                    .terminate_with(TerminationReason::SolverConverged),
                None,
            ));
        }

        let rjtrj = r_n.dot(&r_n);
        let beta = rjtrj / self.rtr;
        self.d = Some(r_n.mul(&float!(-1.0)).add(&d.mul(&beta)));
        self.r = Some(r_n);
        self.p = Some(p_n.clone());
        self.rtr = rjtrj;

        Ok((
            state.param(p_n).cost(self.rtr).gradient(grad).hessian(h),
            None,
        ))
    }

    fn terminate(&mut self, state: &IterState<P, P, (), H, (), F>) -> TerminationStatus {
        if self.r_0_norm < self.epsilon {
            return TerminationStatus::Terminated(TerminationReason::SolverConverged);
        }
        if state.get_iter() >= self.max_iters {
            return TerminationStatus::Terminated(TerminationReason::MaxItersReached);
        }
        TerminationStatus::NotTerminated
    }
}

impl<P, F: ArgminFloat> TrustRegionRadius<F> for Steihaug<P, F> {
    /// Set current radius.
    ///
    /// Needed by [`TrustRegion`](`crate::solver::trustregion::TrustRegion`).
    ///
    /// # Example
    ///
    /// ```
    /// use argmin::solver::trustregion::{Steihaug, TrustRegionRadius};
    /// let mut sh: Steihaug<Vec<f64>, f64> = Steihaug::new();
    /// sh.set_radius(0.8);
    /// ```
    fn set_radius(&mut self, radius: F) {
        self.radius = radius;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::test_utils::TestProblem;
    use crate::core::ArgminError;
    use approx::assert_relative_eq;

    test_trait_impl!(steihaug, Steihaug<TestProblem, f64>);

    #[test]
    fn test_new() {
        let sh: Steihaug<Vec<f64>, f64> = Steihaug::new();

        let Steihaug {
            radius,
            epsilon,
            p,
            r,
            rtr,
            r_0_norm,
            d,
            max_iters,
        } = sh;

        assert_eq!(radius.to_ne_bytes(), f64::NAN.to_ne_bytes());
        assert_eq!(epsilon.to_ne_bytes(), 10e-10f64.to_ne_bytes());
        assert!(p.is_none());
        assert!(r.is_none());
        assert_eq!(rtr.to_ne_bytes(), f64::NAN.to_ne_bytes());
        assert_eq!(r_0_norm.to_ne_bytes(), f64::NAN.to_ne_bytes());
        assert!(d.is_none());
        assert_eq!(max_iters, u64::MAX);
    }

    #[test]
    fn test_with_tolerance() {
        for tolerance in [f64::EPSILON, 1e-10, 1e-12, 1e-6, 1.0, 10.0, 100.0] {
            let sh: Steihaug<Vec<f64>, f64> = Steihaug::new().with_epsilon(tolerance).unwrap();
            assert_eq!(sh.epsilon.to_ne_bytes(), tolerance.to_ne_bytes());
        }

        for tolerance in [-f64::EPSILON, 0.0, -1.0] {
            let res: Result<Steihaug<Vec<f64>, f64>, _> = Steihaug::new().with_epsilon(tolerance);
            assert_error!(
                res,
                ArgminError,
                "Invalid parameter: \"`Steihaug`: epsilon must be > 0.0.\""
            );
        }
    }

    #[test]
    fn test_max_iters() {
        let sh: Steihaug<Vec<f64>, f64> = Steihaug::new();

        let Steihaug { max_iters, .. } = sh;

        assert_eq!(max_iters, u64::MAX);

        for iters in [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144] {
            let sh: Steihaug<Vec<f64>, f64> = Steihaug::new().with_max_iters(iters);

            let Steihaug { max_iters, .. } = sh;

            assert_eq!(max_iters, iters);
        }
    }

    #[test]
    fn test_init() {
        let grad: Vec<f64> = vec![1.0, 2.0];
        let hessian: Vec<Vec<f64>> = vec![vec![4.0, 3.0], vec![2.0, 1.0]];

        let mut sh: Steihaug<Vec<f64>, f64> = Steihaug::new();
        sh.set_radius(1.0);

        // Forgot to initialize gradient
        let state: IterState<Vec<f64>, Vec<f64>, (), Vec<Vec<f64>>, (), f64> = IterState::new();
        let problem = TestProblem::new();
        let res = sh.init(&mut Problem::new(problem), state);
        assert_error!(
            res,
            ArgminError,
            concat!(
                "Not initialized: \"`Steihaug` requires an initial gradient. Please ",
                "provide an initial gradient via `Executor`s `configure` method.\""
            )
        );

        // Forgot to initialize Hessian
        let state: IterState<Vec<f64>, Vec<f64>, (), Vec<Vec<f64>>, (), f64> =
            IterState::new().gradient(grad.clone());
        let problem = TestProblem::new();
        let res = sh.init(&mut Problem::new(problem), state);
        assert_error!(
            res,
            ArgminError,
            concat!(
                "Not initialized: \"`Steihaug` requires an initial Hessian. Please ",
                "provide an initial Hessian via `Executor`s `configure` method.\""
            )
        );

        // All good.
        let state: IterState<Vec<f64>, Vec<f64>, (), Vec<Vec<f64>>, (), f64> =
            IterState::new().gradient(grad.clone()).hessian(hessian);
        let problem = TestProblem::new();
        let (mut state_out, kv) = sh.init(&mut Problem::new(problem), state).unwrap();

        assert!(kv.is_none());

        let s_param = state_out.take_param().unwrap();

        assert_relative_eq!(s_param[0], 0.0f64.sqrt(), epsilon = f64::EPSILON);
        assert_relative_eq!(s_param[1], 0.0f64.sqrt(), epsilon = f64::EPSILON);

        let Steihaug {
            radius,
            epsilon,
            p,
            r,
            rtr,
            r_0_norm,
            d,
            max_iters,
        } = sh;

        assert_eq!(radius.to_ne_bytes(), 1.0f64.to_ne_bytes());
        assert_eq!(epsilon.to_ne_bytes(), 10e-10f64.to_ne_bytes());
        assert_relative_eq!(p.as_ref().unwrap()[0], 0.0f64, epsilon = f64::EPSILON);
        assert_relative_eq!(p.as_ref().unwrap()[1], 0.0f64, epsilon = f64::EPSILON);
        assert_relative_eq!(r.as_ref().unwrap()[0], grad[0], epsilon = f64::EPSILON);
        assert_relative_eq!(r.as_ref().unwrap()[1], grad[1], epsilon = f64::EPSILON);
        assert_eq!(rtr.to_ne_bytes(), 5.0f64.to_ne_bytes());
        assert_eq!(r_0_norm.to_ne_bytes(), (5.0f64).sqrt().to_ne_bytes());
        assert_relative_eq!(d.as_ref().unwrap()[0], -grad[0], epsilon = f64::EPSILON);
        assert_relative_eq!(d.as_ref().unwrap()[1], -grad[1], epsilon = f64::EPSILON);
        assert_eq!(max_iters, u64::MAX);
    }
}
