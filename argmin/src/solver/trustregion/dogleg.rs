// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::{
    ArgminFloat, Error, Gradient, Hessian, IterState, Problem, Solver, State, TerminationReason,
    TrustRegionRadius, KV,
};
use argmin_math::{
    ArgminAdd, ArgminDot, ArgminInv, ArgminL2Norm, ArgminMul, ArgminSub, ArgminWeightedDot,
};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// # Dogleg method
///
/// The Dogleg method computes the intersection of the trust region boundary with a path given by
/// the unconstraind minimum along the steepest descent direction and the optimum of the quadratic
/// approximation of the cost function at the current point.
///
/// ## Requirements on the optimization problem
///
/// The optimization problem is required to implement [`Gradient`] and [`Hessian`].
///
/// ## Reference
///
/// Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(Clone, Debug, Copy, PartialEq, Eq, PartialOrd, Default)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct Dogleg<F> {
    /// Radius
    radius: F,
}

impl<F> Dogleg<F>
where
    F: ArgminFloat,
{
    /// Construct a new instance of [`Dogleg`]
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::trustregion::Dogleg;
    /// let dl: Dogleg<f64> = Dogleg::new();
    /// ```
    pub fn new() -> Self {
        Dogleg { radius: F::nan() }
    }
}

impl<O, F, P, H> Solver<O, IterState<P, P, (), H, F>> for Dogleg<F>
where
    O: Gradient<Param = P, Gradient = P> + Hessian<Param = P, Hessian = H>,
    P: Clone
        + ArgminMul<F, P>
        + ArgminL2Norm<F>
        + ArgminDot<P, F>
        + ArgminAdd<P, P>
        + ArgminSub<P, P>,
    H: ArgminInv<H> + ArgminDot<P, P>,
    F: ArgminFloat,
{
    const NAME: &'static str = "Dogleg";

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        mut state: IterState<P, P, (), H, F>,
    ) -> Result<(IterState<P, P, (), H, F>, Option<KV>), Error> {
        let param = state.take_param().ok_or_else(argmin_error_closure!(
            NotInitialized,
            concat!(
                "`Dogleg` requires an initial parameter vector. ",
                "Please provide an initial guess via `Executor`s `configure` method."
            )
        ))?;

        let g = state
            .take_gradient()
            .map(Result::Ok)
            .unwrap_or_else(|| problem.gradient(&param))?;

        let h = state
            .take_hessian()
            .map(Result::Ok)
            .unwrap_or_else(|| problem.hessian(&param))?;

        let pstar;

        // pb = -H^-1g
        let pb = (h.inv()?).dot(&g).mul(&float!(-1.0));

        if pb.l2_norm() <= self.radius {
            pstar = pb;
        } else {
            // pu = - (g^Tg)/(g^THg) * g
            let pu = g.mul(&(-g.dot(&g) / g.weighted_dot(&h, &g)));

            let k = pb.sub(&pu); // p^b - p^u
            let c = pu.dot(&k); // p^u^T * (p^b - p^u)
            let k = k.dot(&k); // (p^b - p^u)^T (p^b - p^u)
            let u = pu.dot(&pu); // p^u^T p^u

            let delta_squared = self.radius.powi(2);
            let t1 = (c.powi(2) + delta_squared * k - k * u).sqrt();
            let tau = [
                -(t1 + c - k) / k,
                (t1 - c + k) / k,
                (float!(2.0) * c + delta_squared - u) / (float!(2.0) * c),
            ]
            .into_iter()
            // .enumerate()
            // .map(|(i, t)| {
            //     println!("tau{}: {}", i, t);
            //     t
            // })
            .filter(|t| !t.is_nan() && !t.is_infinite() && *t >= float!(0.0) && *t <= float!(2.0))
            .fold(float!(0.0), |acc, t| if t >= acc { t } else { acc });

            if tau >= float!(0.0) && tau < float!(1.0) {
                pstar = pu.mul(&tau);
            } else if tau >= float!(1.0) && tau <= float!(2.0) {
                pstar = pu.add(&pb.sub(&pu).mul(&(tau - float!(1.0))));
            } else {
                return Err(argmin_error!(
                    PotentialBug,
                    "tau is outside the range [0, 2], this is not supposed to happen."
                ));
            }
        }
        Ok((state.param(pstar).gradient(g).hessian(h), None))
    }

    fn terminate(&mut self, state: &IterState<P, P, (), H, F>) -> TerminationReason {
        if state.get_iter() >= 1 {
            TerminationReason::MaxItersReached
        } else {
            TerminationReason::NotTerminated
        }
    }
}

impl<F: ArgminFloat> TrustRegionRadius<F> for Dogleg<F> {
    /// Set current radius.
    ///
    /// Needed by [`TrustRegion`](`crate::solver::trustregion::TrustRegion`).
    ///
    /// # Example
    ///
    /// ```
    /// use argmin::solver::trustregion::{Dogleg, TrustRegionRadius};
    /// let mut dl: Dogleg<f64> = Dogleg::new();
    /// dl.set_radius(0.8);
    /// ```
    fn set_radius(&mut self, radius: F) {
        self.radius = radius;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "_ndarrayl")]
    use crate::core::ArgminError;
    use crate::test_trait_impl;

    test_trait_impl!(dogleg, Dogleg<f64>);

    #[test]
    fn test_new() {
        let dl: Dogleg<f64> = Dogleg::new();

        let Dogleg { radius } = dl;

        assert_eq!(radius.to_ne_bytes(), f64::NAN.to_ne_bytes());
    }

    #[cfg(feature = "_ndarrayl")]
    #[test]
    fn test_next_iter() {
        use approx::assert_relative_eq;
        use ndarray::{Array, Array1, Array2};

        struct TestProblem {}

        impl Gradient for TestProblem {
            type Param = Array1<f64>;
            type Gradient = Array1<f64>;

            fn gradient(&self, _p: &Self::Param) -> Result<Self::Gradient, Error> {
                Ok(Array1::from_vec(vec![0.5, 2.0]))
            }
        }

        impl Hessian for TestProblem {
            type Param = Array1<f64>;
            type Hessian = Array2<f64>;

            fn hessian(&self, _p: &Self::Param) -> Result<Self::Hessian, Error> {
                Ok(Array::from_shape_vec((2, 2), vec![1f64, 2.0, 3.0, 4.0])?)
            }
        }

        let param: Array1<f64> = Array1::from_vec(vec![-1.0, 1.0]);

        let mut dl: Dogleg<f64> = Dogleg::new();
        dl.set_radius(1.0);

        // Forgot to initialize the parameter vector
        let state: IterState<Array1<f64>, Array1<f64>, (), Array2<f64>, f64> = IterState::new();
        let problem = TestProblem {};
        let res = dl.next_iter(&mut Problem::new(problem), state);
        assert_error!(
            res,
            ArgminError,
            concat!(
                "Not initialized: \"`Dogleg` requires an initial parameter vector. Please ",
                "provide an initial guess via `Executor`s `configure` method.\""
            )
        );

        // All good.
        let state: IterState<Array1<f64>, Array1<f64>, (), Array2<f64>, f64> =
            IterState::new().param(param);
        let problem = TestProblem {};
        let (mut state_out, kv) = dl.next_iter(&mut Problem::new(problem), state).unwrap();

        assert!(kv.is_none());

        let s_param = state_out.take_param().unwrap();

        assert_relative_eq!(s_param[0], -0.9730617585026127, epsilon = f64::EPSILON);
        assert_relative_eq!(s_param[1], 0.2305446033629983, epsilon = f64::EPSILON);
    }
}
