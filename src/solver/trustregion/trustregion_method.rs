// Copyright 2018-2020 argmin developers
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
use crate::solver::trustregion::reduction_ratio;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

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
/// [Example](https://github.com/argmin-rs/argmin/blob/master/examples/trustregion_nd.rs)
///
/// # References:
///
/// [0] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(Clone, Serialize, Deserialize)]
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

impl<R, F: ArgminFloat> TrustRegion<R, F> {
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
    pub fn radius(mut self, radius: F) -> Self {
        self.radius = radius;
        self
    }

    /// Set maximum radius
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

impl<O, R, F> Solver<O, F> for TrustRegion<R, F>
where
    O: ArgminOp<Output = F>,
    O::Param: Default
        + Clone
        + Debug
        + Serialize
        + ArgminMul<F, O::Param>
        + ArgminWeightedDot<O::Param, F, O::Hessian>
        + ArgminNorm<F>
        + ArgminDot<O::Param, F>
        + ArgminAdd<O::Param, O::Param>
        + ArgminSub<O::Param, O::Param>
        + ArgminZeroLike
        + ArgminMul<F, O::Param>,
    O::Hessian: Default + Clone + Debug + Serialize + ArgminDot<O::Param, O::Param>,
    R: ArgminTrustRegion<F> + Solver<OpWrapper<O>, F>,
    F: ArgminFloat,
{
    const NAME: &'static str = "Trust region";

    fn init(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &IterState<O, F>,
    ) -> Result<Option<ArgminIterData<O, F>>, Error> {
        let param = state.get_param();
        let grad = op.gradient(&param)?;
        let hessian = op.hessian(&param)?;
        self.fxk = op.apply(&param)?;
        self.mk0 = self.fxk;
        Ok(Some(
            ArgminIterData::new()
                .param(param)
                .cost(self.fxk)
                .grad(grad)
                .hessian(hessian),
        ))
    }

    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &IterState<O, F>,
    ) -> Result<ArgminIterData<O, F>, Error> {
        let param = state.get_param();
        let grad = state
            .get_grad()
            .unwrap_or_else(|| op.gradient(&param).unwrap());
        let hessian = state
            .get_hessian()
            .unwrap_or_else(|| op.hessian(&param).unwrap());

        self.subproblem.set_radius(self.radius);

        let ArgminResult {
            operator: sub_op,
            state: IterState { param: pk, .. },
        } = Executor::new(
            OpWrapper::new_from_wrapper(op),
            self.subproblem.clone(),
            param.clone(),
        )
        .grad(grad.clone())
        .hessian(hessian.clone())
        .ctrlc(false)
        .run()?;

        // Operator must be consumed again, otherwise the operator, which moved into the subproblem
        // executor as well as the function evaluation counts are lost.
        op.consume_op(sub_op);

        let new_param = pk.add(&param);
        let fxkpk = op.apply(&new_param)?;
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

        Ok(if rho > self.eta {
            self.fxk = fxkpk;
            self.mk0 = fxkpk;
            let grad = op.gradient(&new_param)?;
            let hessian = op.hessian(&new_param)?;
            ArgminIterData::new()
                .param(new_param)
                .cost(fxkpk)
                .grad(grad)
                .hessian(hessian)
        } else {
            ArgminIterData::new().param(param).cost(self.fxk)
        }
        .kv(make_kv!("radius" => cur_radius;)))
    }

    fn terminate(&mut self, _state: &IterState<O, F>) -> TerminationReason {
        // todo
        TerminationReason::NotTerminated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::trustregion::steihaug::Steihaug;
    use crate::test_trait_impl;

    type Operator = MinimalNoOperator;

    test_trait_impl!(trustregion, TrustRegion<Steihaug<Operator, f64>, f64>);
}
