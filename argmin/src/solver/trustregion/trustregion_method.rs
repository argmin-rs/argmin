// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::{
    ArgminFloat, CostFunction, Error, Executor, Gradient, Hessian, IterState, OptimizationResult,
    Problem, Solver, TerminationStatus, TrustRegionRadius, KV,
};
use crate::solver::trustregion::reduction_ratio;
use argmin_math::{ArgminAdd, ArgminDot, ArgminL2Norm, ArgminWeightedDot};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// # Trust region method
///
/// The trust region method approximates the cost function within a certain region around the
/// current point in parameter space. Depending on the quality of this approximation, the region is
/// either expanded or contracted.
///
/// The calculation of the actual step length and direction is performed by a method which
/// implements [`TrustRegionRadius`](`crate::solver::trustregion::TrustRegionRadius`), such as:
///
/// * [Cauchy point](`crate::solver::trustregion::CauchyPoint`)
/// * [Dogleg method](`crate::solver::trustregion::Dogleg`)
/// * [Steihaug method](`crate::solver::trustregion::Steihaug`)
///
/// ## Requirements on the optimization problem
///
/// The optimization problem is required to implement [`CostFunction`], [`Gradient`] and
/// [`Hessian`].
///
/// ## Reference
///
/// Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct TrustRegion<R, F> {
    /// Radius
    radius: F,
    /// Maximum radius
    max_radius: F,
    /// eta \in [0, 1/4)
    eta: F,
    /// subproblem (must implement [`crate::solver::trustregion::TrustRegionRadius`])
    subproblem: R,
    /// f(xk)
    fxk: F,
    /// mk(0)
    mk0: F,
}

impl<R, F> TrustRegion<R, F>
where
    F: ArgminFloat,
{
    /// Construct a new instance of [`TrustRegion`]
    ///
    /// # Example
    ///
    /// ```
    /// use argmin::solver::trustregion::{CauchyPoint, TrustRegion};
    /// let cp: CauchyPoint<f64> = CauchyPoint::new();
    /// let tr: TrustRegion<_, f64> = TrustRegion::new(cp);
    /// ```
    pub fn new(subproblem: R) -> Self {
        TrustRegion {
            radius: float!(1.0),
            max_radius: float!(100.0),
            eta: float!(0.125),
            subproblem,
            fxk: F::nan(),
            mk0: F::nan(),
        }
    }

    /// Set radius
    ///
    /// Defaults to `1.0`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::trustregion::{TrustRegion, CauchyPoint};
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// let cp: CauchyPoint<f64> = CauchyPoint::new();
    /// let tr: TrustRegion<_, f64> = TrustRegion::new(cp).with_radius(0.8)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_radius(mut self, radius: F) -> Result<Self, Error> {
        if radius <= float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`TrustRegion`: radius must be > 0."
            ));
        }
        self.radius = radius;
        Ok(self)
    }

    /// Set maximum radius
    ///
    /// Defaults to `100.0`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::trustregion::{TrustRegion, CauchyPoint};
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// let cp: CauchyPoint<f64> = CauchyPoint::new();
    /// let tr: TrustRegion<_, f64> = TrustRegion::new(cp).with_max_radius(1000.0)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_max_radius(mut self, max_radius: F) -> Result<Self, Error> {
        if max_radius <= float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`TrustRegion`: maximum radius must be > 0."
            ));
        }
        self.max_radius = max_radius;
        Ok(self)
    }

    /// Set eta
    ///
    /// Must lie in `[0, 1/4)` and defaults to `0.125`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::trustregion::{TrustRegion, CauchyPoint};
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// let cp: CauchyPoint<f64> = CauchyPoint::new();
    /// let tr: TrustRegion<_, f64> = TrustRegion::new(cp).with_eta(0.2)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_eta(mut self, eta: F) -> Result<Self, Error> {
        if eta >= float!(0.25) || eta < float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`TrustRegion`: eta must be in [0, 1/4)."
            ));
        }
        self.eta = eta;
        Ok(self)
    }
}

impl<O, R, F, P, G, H> Solver<O, IterState<P, G, (), H, (), F>> for TrustRegion<R, F>
where
    O: CostFunction<Param = P, Output = F>
        + Gradient<Param = P, Gradient = G>
        + Hessian<Param = P, Hessian = H>,
    P: Clone + ArgminL2Norm<F> + ArgminDot<P, F> + ArgminDot<G, F> + ArgminAdd<P, P>,
    G: Clone,
    H: Clone + ArgminDot<P, P>,
    R: Clone + TrustRegionRadius<F> + Solver<O, IterState<P, G, (), H, (), F>>,
    F: ArgminFloat,
{
    const NAME: &'static str = "Trust region";

    fn init(
        &mut self,
        problem: &mut Problem<O>,
        mut state: IterState<P, G, (), H, (), F>,
    ) -> Result<(IterState<P, G, (), H, (), F>, Option<KV>), Error> {
        let param = state.take_param().ok_or_else(argmin_error_closure!(
            NotInitialized,
            concat!(
                "`TrustRegion` requires an initial parameter vector. ",
                "Please provide an initial guess via `Executor`s `configure` method."
            )
        ))?;

        let grad = state
            .take_gradient()
            .map(Result::Ok)
            .unwrap_or_else(|| problem.gradient(&param))?;

        let hessian = state
            .take_hessian()
            .map(Result::Ok)
            .unwrap_or_else(|| problem.hessian(&param))?;

        let cost = state.get_cost();
        self.fxk = if cost.is_infinite() && cost.is_sign_positive() {
            problem.cost(&param)?
        } else {
            cost
        };

        self.mk0 = self.fxk;
        Ok((
            state
                .param(param)
                .cost(self.fxk)
                .gradient(grad)
                .hessian(hessian),
            None,
        ))
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        mut state: IterState<P, G, (), H, (), F>,
    ) -> Result<(IterState<P, G, (), H, (), F>, Option<KV>), Error> {
        let param = state.take_param().ok_or_else(argmin_error_closure!(
            PotentialBug,
            "`TrustRegion`: Parameter vector in state not set."
        ))?;

        let grad = state.take_gradient().ok_or_else(argmin_error_closure!(
            PotentialBug,
            "`TrustRegion`: Gradient in state not set."
        ))?;

        let hessian = state.take_hessian().ok_or_else(argmin_error_closure!(
            PotentialBug,
            "`TrustRegion`: Hessian in state not set."
        ))?;

        self.subproblem.set_radius(self.radius);

        let OptimizationResult {
            problem: sub_problem,
            state: mut sub_state,
            ..
        } = Executor::new(problem.take_problem().unwrap(), self.subproblem.clone())
            .configure(|config| {
                config
                    .param(param.clone())
                    .gradient(grad.clone())
                    .hessian(hessian.clone())
            })
            .ctrlc(false)
            .run()?;

        let pk = sub_state.take_param().unwrap();

        // Consume intermediate problem again. This takes care of the function evaluation counts.
        problem.consume_problem(sub_problem);

        let new_param = pk.add(&param);
        let fxkpk = problem.cost(&new_param)?;
        let mkpk = self.fxk + pk.dot(&grad) + float!(0.5) * pk.weighted_dot(&hessian, &pk);

        let rho = reduction_ratio(self.fxk, fxkpk, self.mk0, mkpk);

        let pk_norm = pk.l2_norm();

        let cur_radius = self.radius;

        self.radius = if rho < float!(0.25) {
            float!(0.25) * pk_norm
        } else if rho > float!(0.75) && (pk_norm - self.radius).abs() <= float!(10.0) * F::epsilon()
        {
            self.max_radius.min(float!(2.0) * self.radius)
        } else {
            self.radius
        };

        Ok((
            if rho > self.eta {
                self.fxk = fxkpk;
                self.mk0 = fxkpk;
                let grad = problem.gradient(&new_param)?;
                let hessian = problem.hessian(&new_param)?;
                state
                    .param(new_param)
                    .cost(fxkpk)
                    .gradient(grad)
                    .hessian(hessian)
            } else {
                state
                    .param(param)
                    .cost(self.fxk)
                    .gradient(grad)
                    .hessian(hessian)
            },
            Some(kv!("radius" => cur_radius;)),
        ))
    }

    fn terminate(&mut self, _state: &IterState<P, G, (), H, (), F>) -> TerminationStatus {
        TerminationStatus::NotTerminated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::test_utils::TestProblem;
    use crate::core::{ArgminError, State};
    use crate::solver::trustregion::{CauchyPoint, Steihaug};
    use crate::test_trait_impl;

    test_trait_impl!(trustregion, TrustRegion<Steihaug<TestProblem, f64>, f64>);

    #[test]
    fn test_new() {
        let cp: CauchyPoint<f64> = CauchyPoint::new();
        let tr: TrustRegion<_, f64> = TrustRegion::new(cp);

        let TrustRegion {
            radius,
            max_radius,
            eta,
            subproblem: _,
            fxk,
            mk0,
        } = tr;

        assert_eq!(radius.to_ne_bytes(), 1.0f64.to_ne_bytes());
        assert_eq!(max_radius.to_ne_bytes(), 100.0f64.to_ne_bytes());
        assert_eq!(eta.to_ne_bytes(), 0.125f64.to_ne_bytes());
        assert_eq!(fxk.to_ne_bytes(), f64::NAN.to_ne_bytes());
        assert_eq!(mk0.to_ne_bytes(), f64::NAN.to_ne_bytes());
    }

    #[test]
    fn test_with_radius() {
        // correct parameters
        for radius in [std::f64::EPSILON, 1e-2, 1.0, 2.0, 10.0, 100.0] {
            let cp: CauchyPoint<f64> = CauchyPoint::new();
            let tr: TrustRegion<_, f64> = TrustRegion::new(cp);
            let res = tr.with_radius(radius);
            assert!(res.is_ok());

            let nm = res.unwrap();
            assert_eq!(nm.radius.to_ne_bytes(), radius.to_ne_bytes());
        }

        // incorrect parameters
        for radius in [0.0, -f64::EPSILON, -1.0, -100.0, -42.0] {
            let cp: CauchyPoint<f64> = CauchyPoint::new();
            let tr: TrustRegion<_, f64> = TrustRegion::new(cp);
            let res = tr.with_radius(radius);
            assert_error!(
                res,
                ArgminError,
                "Invalid parameter: \"`TrustRegion`: radius must be > 0.\""
            );
        }
    }

    #[test]
    fn test_with_eta() {
        // correct parameters
        for eta in [
            0.0,
            std::f64::EPSILON,
            1e-2,
            0.125,
            0.25 - std::f64::EPSILON,
        ] {
            let cp: CauchyPoint<f64> = CauchyPoint::new();
            let tr: TrustRegion<_, f64> = TrustRegion::new(cp);
            let res = tr.with_eta(eta);
            assert!(res.is_ok());

            let nm = res.unwrap();
            assert_eq!(nm.eta.to_ne_bytes(), eta.to_ne_bytes());
        }

        // incorrect parameters
        for eta in [-f64::EPSILON, -1.0, -100.0, -42.0, 0.25, 1.0] {
            let cp: CauchyPoint<f64> = CauchyPoint::new();
            let tr: TrustRegion<_, f64> = TrustRegion::new(cp);
            let res = tr.with_eta(eta);
            assert_error!(
                res,
                ArgminError,
                "Invalid parameter: \"`TrustRegion`: eta must be in [0, 1/4).\""
            );
        }
    }

    #[test]
    fn test_init() {
        let param: Vec<f64> = vec![1.0, 2.0];

        let cp: CauchyPoint<f64> = CauchyPoint::new();
        let mut tr: TrustRegion<_, f64> = TrustRegion::new(cp);

        // Forgot to initialize parameter vector
        let state: IterState<Vec<f64>, Vec<f64>, (), Vec<Vec<f64>>, (), f64> = IterState::new();
        let problem = TestProblem::new();
        let res = tr.init(&mut Problem::new(problem), state);
        assert_error!(
            res,
            ArgminError,
            concat!(
                "Not initialized: \"`TrustRegion` requires an initial parameter vector. Please ",
                "provide an initial guess via `Executor`s `configure` method.\""
            )
        );

        // All good.
        let state: IterState<Vec<f64>, Vec<f64>, (), Vec<Vec<f64>>, (), f64> =
            IterState::new().param(param.clone());
        let problem = TestProblem::new();
        let (mut state_out, kv) = tr.init(&mut Problem::new(problem), state).unwrap();

        assert!(kv.is_none());

        let s_param = state_out.take_param().unwrap();

        assert_eq!(s_param[0].to_ne_bytes(), param[0].to_ne_bytes());
        assert_eq!(s_param[1].to_ne_bytes(), param[1].to_ne_bytes());

        let TrustRegion {
            radius,
            max_radius,
            eta,
            subproblem: _,
            fxk,
            mk0,
        } = tr;

        assert_eq!(radius.to_ne_bytes(), 1.0f64.to_ne_bytes());
        assert_eq!(max_radius.to_ne_bytes(), 100.0f64.to_ne_bytes());
        assert_eq!(eta.to_ne_bytes(), 0.125f64.to_ne_bytes());
        assert_eq!(fxk.to_ne_bytes(), 1.0f64.sqrt().to_ne_bytes());
        assert_eq!(mk0.to_ne_bytes(), 1.0f64.to_ne_bytes());
    }
}
