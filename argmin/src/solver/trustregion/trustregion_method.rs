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
    ArgminError, ArgminFloat, CostFunction, DeserializeOwnedAlias, Error, Executor, Gradient,
    Hessian, IterState, OptimizationResult, Problem, SerializeAlias, Solver, TerminationReason,
    TrustRegionRadius, KV,
};
use crate::solver::trustregion::reduction_ratio;
use argmin_math::{ArgminAdd, ArgminDot, ArgminNorm, ArgminWeightedDot};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// The trust region method approximates the cost function within a certain region around the
/// current point in parameter space. Depending on the quality of this approximation, the region is
/// either expanded or contracted.
///
/// The calculation of the actual step length and direction is done by one of the following
/// methods:
///
/// * [Cauchy point](../cauchypoint/struct.CauchyPoint.html)
/// * [Dogleg method](../dogleg/struct.Dogleg.html)
/// * [Steihaug method](../steihaug/struct.Steihaug.html)
///
/// This subproblem can be set via `set_subproblem(...)`. If this is not provided, it will default
/// to the Steihaug method.
///
/// # References:
///
/// \[0\] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct TrustRegion<R, F> {
    /// Radius
    radius: F,
    /// Maximum Radius
    max_radius: F,
    /// eta \in [0, 1/4)
    eta: F,
    /// subproblem
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
    /// Constructor
    pub fn new(subproblem: R) -> Self {
        TrustRegion {
            radius: F::from_f64(1.0).unwrap(),
            max_radius: F::from_f64(100.0).unwrap(),
            eta: F::from_f64(0.125).unwrap(),
            subproblem,
            fxk: F::nan(),
            mk0: F::nan(),
        }
    }

    /// set radius
    #[must_use]
    pub fn radius(mut self, radius: F) -> Self {
        self.radius = radius;
        self
    }

    /// Set maximum radius
    #[must_use]
    pub fn max_radius(mut self, max_radius: F) -> Self {
        self.max_radius = max_radius;
        self
    }

    /// Set eta
    pub fn eta(mut self, eta: F) -> Result<Self, Error> {
        if eta >= F::from_f64(0.25).unwrap() || eta < F::from_f64(0.0).unwrap() {
            return Err(ArgminError::InvalidParameter {
                text: "TrustRegion: eta must be in [0, 1/4).".to_string(),
            }
            .into());
        }
        self.eta = eta;
        Ok(self)
    }
}

impl<O, R, F, P, G, H> Solver<O, IterState<P, G, (), H, F>> for TrustRegion<R, F>
where
    O: CostFunction<Param = P, Output = F>
        + Gradient<Param = P, Gradient = G>
        + Hessian<Param = P, Hessian = H>,
    P: Clone
        + SerializeAlias
        + DeserializeOwnedAlias
        + ArgminNorm<F>
        + ArgminDot<P, F>
        + ArgminDot<G, F>
        + ArgminAdd<P, P>,
    G: Clone + SerializeAlias + DeserializeOwnedAlias,
    H: Clone + SerializeAlias + DeserializeOwnedAlias + ArgminDot<P, P>,
    R: Clone + TrustRegionRadius<F> + Solver<O, IterState<P, G, (), H, F>>,
    F: ArgminFloat,
{
    const NAME: &'static str = "Trust region";

    fn init(
        &mut self,
        problem: &mut Problem<O>,
        mut state: IterState<P, G, (), H, F>,
    ) -> Result<(IterState<P, G, (), H, F>, Option<KV>), Error> {
        let param = state.take_param().unwrap();
        let grad = problem.gradient(&param)?;
        let hessian = problem.hessian(&param)?;
        self.fxk = problem.cost(&param)?;
        self.mk0 = self.fxk;
        Ok((
            state
                .param(param)
                .cost(self.fxk)
                .grad(grad)
                .hessian(hessian),
            None,
        ))
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        mut state: IterState<P, G, (), H, F>,
    ) -> Result<(IterState<P, G, (), H, F>, Option<KV>), Error> {
        let param = state.take_param().unwrap();
        let grad = state
            .take_grad()
            .map(Result::Ok)
            .unwrap_or_else(|| problem.gradient(&param))?;
        let hessian = state
            .take_hessian()
            .map(Result::Ok)
            .unwrap_or_else(|| problem.hessian(&param))?;

        self.subproblem.set_radius(self.radius);

        let OptimizationResult {
            problem: sub_problem,
            state: mut sub_state,
            ..
        } = Executor::new(problem.take_problem().unwrap(), self.subproblem.clone())
            .configure(|config| {
                config
                    .param(param.clone())
                    .grad(grad.clone())
                    .hessian(hessian.clone())
            })
            .ctrlc(false)
            .run()?;

        let pk = sub_state.take_param().unwrap();

        // Consume intermediate problem again. This takes care of the function evaluation counts.
        problem.consume_problem(sub_problem);

        let new_param = pk.add(&param);
        let fxkpk = problem.cost(&new_param)?;
        let mkpk =
            self.fxk + pk.dot(&grad) + F::from_f64(0.5).unwrap() * pk.weighted_dot(&hessian, &pk);

        let rho = reduction_ratio(self.fxk, fxkpk, self.mk0, mkpk);

        let pk_norm = pk.norm();

        let cur_radius = self.radius;
        self.radius = if rho < F::from_f64(0.25).unwrap() {
            F::from_f64(0.25).unwrap() * pk_norm
        } else if rho > F::from_f64(0.75).unwrap()
            && (pk_norm - self.radius).abs() <= F::from_f64(10.0).unwrap() * F::epsilon()
        {
            self.max_radius.min(F::from_f64(2.0).unwrap() * self.radius)
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
                    .grad(grad)
                    .hessian(hessian)
            } else {
                state
                    .param(param)
                    .cost(self.fxk)
                    .grad(grad)
                    .hessian(hessian)
            },
            Some(make_kv!("radius" => cur_radius;)),
        ))
    }

    fn terminate(&mut self, _state: &IterState<P, G, (), H, F>) -> TerminationReason {
        // todo
        TerminationReason::NotTerminated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::test_utils::TestProblem;
    use crate::solver::trustregion::steihaug::Steihaug;
    use crate::test_trait_impl;

    test_trait_impl!(trustregion, TrustRegion<Steihaug<TestProblem, f64>, f64>);
}
