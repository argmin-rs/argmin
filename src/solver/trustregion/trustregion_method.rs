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
pub struct TrustRegion<R> {
    /// Radius
    radius: f64,
    /// Maximum Radius
    max_radius: f64,
    /// eta \in [0, 1/4)
    eta: f64,
    /// subproblem
    subproblem: R,
    /// f(xk)
    fxk: f64,
    /// mk(0)
    mk0: f64,
}

impl<R> TrustRegion<R> where {
    /// Constructor
    pub fn new(subproblem: R) -> Self {
        TrustRegion {
            radius: 1.0,
            max_radius: 100.0,
            eta: 0.125,
            subproblem,
            fxk: std::f64::NAN,
            mk0: std::f64::NAN,
        }
    }

    /// set radius
    pub fn radius(mut self, radius: f64) -> Self {
        self.radius = radius;
        self
    }

    /// Set maximum radius
    pub fn max_radius(mut self, max_radius: f64) -> Self {
        self.max_radius = max_radius;
        self
    }

    /// Set eta
    pub fn eta(mut self, eta: f64) -> Result<Self, Error> {
        if eta >= 0.25 || eta < 0.0 {
            return Err(ArgminError::InvalidParameter {
                text: "TrustRegion: eta must be in [0, 1/4).".to_string(),
            }
            .into());
        }
        self.eta = eta;
        Ok(self)
    }
}

impl<O, R> Solver<O> for TrustRegion<R>
where
    O: ArgminOp<Output = f64>,
    O::Param: Default
        + Clone
        + Debug
        + Serialize
        + ArgminMul<f64, O::Param>
        + ArgminWeightedDot<O::Param, f64, O::Hessian>
        + ArgminNorm<f64>
        + ArgminDot<O::Param, f64>
        + ArgminAdd<O::Param, O::Param>
        + ArgminSub<O::Param, O::Param>
        + ArgminZeroLike
        + ArgminMul<f64, O::Param>,
    O::Hessian: Default + Clone + Debug + Serialize + ArgminDot<O::Param, O::Param>,
    R: ArgminTrustRegion + Solver<OpWrapper<O>>,
{
    const NAME: &'static str = "Trust region";

    fn init(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &IterState<O>,
    ) -> Result<Option<ArgminIterData<O>>, Error> {
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
        state: &IterState<O>,
    ) -> Result<ArgminIterData<O>, Error> {
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
            OpWrapper::new_from_op(&op),
            self.subproblem.clone(),
            param.clone(),
        )
        .grad(grad.clone())
        .hessian(hessian.clone())
        .ctrlc(false)
        .run()?;

        op.consume_op(sub_op);

        let new_param = pk.add(&param);
        let fxkpk = op.apply(&new_param)?;
        let mkpk = self.fxk + pk.dot(&grad) + 0.5 * pk.weighted_dot(&hessian, &pk);

        let rho = reduction_ratio(self.fxk, fxkpk, self.mk0, mkpk);

        let pk_norm = pk.norm();

        let cur_radius = self.radius;
        self.radius = if rho < 0.25 {
            0.25 * pk_norm
        } else if rho > 0.75 && (pk_norm - self.radius).abs() <= 10.0 * std::f64::EPSILON {
            self.max_radius.min(2.0 * self.radius)
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

    fn terminate(&mut self, _state: &IterState<O>) -> TerminationReason {
        // todo
        TerminationReason::NotTerminated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_trait_impl;
    use crate::solver::trustregion::steihaug::Steihaug;

    type Operator = MinimalNoOperator;

    test_trait_impl!(trustregion, TrustRegion<Steihaug<Operator>>);
}
