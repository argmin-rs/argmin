// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::{
    ArgminFloat, Error, Gradient, Hessian, IterState, Problem, Solver, State, TerminationReason,
    TerminationStatus, TrustRegionRadius, KV,
};
use argmin_math::{ArgminL2Norm, ArgminMul, ArgminWeightedDot};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// # Cauchy point method
///
/// The Cauchy point is the minimum of the quadratic approximation of the cost function within the
/// trust region along the direction given by the first derivative.
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
pub struct CauchyPoint<F> {
    /// Radius
    radius: F,
}

impl<F> CauchyPoint<F>
where
    F: ArgminFloat,
{
    /// Construct a new instance of [`CauchyPoint`]
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::trustregion::CauchyPoint;
    /// let cp: CauchyPoint<f64> = CauchyPoint::new();
    /// ```
    pub fn new() -> Self {
        CauchyPoint { radius: F::nan() }
    }
}

impl<O, F, P, G, H> Solver<O, IterState<P, G, (), H, F>> for CauchyPoint<F>
where
    O: Gradient<Param = P, Gradient = G> + Hessian<Param = P, Hessian = H>,
    P: Clone + Debug + ArgminMul<F, P> + ArgminWeightedDot<P, F, H>,
    G: ArgminMul<F, P> + ArgminWeightedDot<G, F, H> + ArgminL2Norm<F>,
    F: ArgminFloat,
{
    const NAME: &'static str = "Cauchy Point";

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        mut state: IterState<P, G, (), H, F>,
    ) -> Result<(IterState<P, G, (), H, F>, Option<KV>), Error> {
        let param = state.take_param().ok_or_else(argmin_error_closure!(
            NotInitialized,
            concat!(
                "`CauchyPoint` requires an initial parameter vector. ",
                "Please provide an initial guess via `Executor`s `configure` method."
            )
        ))?;

        let grad = state
            .take_gradient()
            .map(Result::Ok)
            .unwrap_or_else(|| problem.gradient(&param))?;

        let grad_norm = grad.l2_norm();

        let hessian = state
            .take_hessian()
            .map(Result::Ok)
            .unwrap_or_else(|| problem.hessian(&param))?;

        let wdp = grad.weighted_dot(&hessian, &grad);

        let tau: F = if wdp <= float!(0.0) {
            float!(1.0)
        } else {
            float!(1.0).min(grad_norm.powi(3) / (self.radius * wdp))
        };

        let new_param = grad.mul(&(-tau * self.radius / grad_norm));
        Ok((state.param(new_param), None))
    }

    fn terminate(&mut self, state: &IterState<P, G, (), H, F>) -> TerminationStatus {
        // Not an iterative algorithm
        if state.get_iter() >= 1 {
            TerminationStatus::Terminated(TerminationReason::MaxItersReached)
        } else {
            TerminationStatus::NotTerminated
        }
    }
}

impl<F> TrustRegionRadius<F> for CauchyPoint<F>
where
    F: ArgminFloat,
{
    /// Set current radius.
    ///
    /// Needed by [`TrustRegion`](`crate::solver::trustregion::TrustRegion`).
    ///
    /// # Example
    ///
    /// ```
    /// use argmin::solver::trustregion::{CauchyPoint, TrustRegionRadius};
    /// let mut cp: CauchyPoint<f64> = CauchyPoint::new();
    /// cp.set_radius(0.8);
    /// ```
    fn set_radius(&mut self, radius: F) {
        self.radius = radius;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{test_utils::TestProblem, ArgminError};
    use crate::test_trait_impl;
    use approx::assert_relative_eq;

    test_trait_impl!(cauchypoint, CauchyPoint<f64>);

    #[test]
    fn test_new() {
        let cp: CauchyPoint<f64> = CauchyPoint::new();

        let CauchyPoint { radius } = cp;

        assert_eq!(radius.to_ne_bytes(), f64::NAN.to_ne_bytes());
    }

    #[test]
    fn test_next_iter() {
        let param: Vec<f64> = vec![-1.0, 1.0];

        let mut cp: CauchyPoint<f64> = CauchyPoint::new();
        cp.set_radius(1.0);

        // Forgot to initialize the parameter vector
        let state: IterState<Vec<f64>, Vec<f64>, (), Vec<Vec<f64>>, f64> = IterState::new();
        let problem = TestProblem::new();
        let res = cp.next_iter(&mut Problem::new(problem), state);
        assert_error!(
            res,
            ArgminError,
            concat!(
                "Not initialized: \"`CauchyPoint` requires an initial parameter vector. Please ",
                "provide an initial guess via `Executor`s `configure` method.\""
            )
        );

        // All good.
        let state: IterState<Vec<f64>, Vec<f64>, (), Vec<Vec<f64>>, f64> =
            IterState::new().param(param);
        let problem = TestProblem::new();
        let (mut state_out, kv) = cp.next_iter(&mut Problem::new(problem), state).unwrap();

        assert!(kv.is_none());

        let s_param = state_out.take_param().unwrap();

        assert_relative_eq!(s_param[0], 1.0f64 / 2.0f64.sqrt(), epsilon = f64::EPSILON);
        assert_relative_eq!(s_param[1], -1.0f64 / 2.0f64.sqrt(), epsilon = f64::EPSILON);
    }
}
