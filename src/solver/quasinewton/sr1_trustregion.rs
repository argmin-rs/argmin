// Copyright 2019 Stefan Kroboth
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
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// SR1 Trust Region method
///
/// [Example](https://github.com/argmin-rs/argmin/blob/master/examples/sr1_trustregion.rs)
///
/// # References:
///
/// [0] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(Clone, Serialize, Deserialize)]
pub struct SR1TrustRegion<B, R> {
    /// parameter for skipping rule
    r: f64,
    /// Inverse Hessian
    init_hessian: Option<B>,
    /// subproblem
    subproblem: R,
    /// Radius
    radius: f64,
    // /// Maximum Radius
    // max_radius: f64,
    /// eta \in [0, 1/4)
    eta: f64,
}

impl<B, R> SR1TrustRegion<B, R> {
    /// Constructor
    pub fn new(subproblem: R) -> Self {
        SR1TrustRegion {
            r: 1e-8,
            init_hessian: None,
            subproblem,
            radius: 1.0,
            // max_radius: 100.0,
            eta: 0.5 * 1e-3,
        }
    }

    /// provide initial Hessian (if not provided, the algorithm will try to compute it using the
    /// `hessian` method of `ArgminOp`.
    pub fn hessian(mut self, init_hessian: B) -> Self {
        self.init_hessian = Some(init_hessian);
        self
    }

    /// Set r
    pub fn r(mut self, r: f64) -> Result<Self, Error> {
        if r <= 0.0 || r >= 1.0 {
            Err(ArgminError::InvalidParameter {
                text: "SR1: r must be in (0, 1).".to_string(),
            }
            .into())
        } else {
            self.r = r;
            Ok(self)
        }
    }

    /// set radius
    pub fn radius(mut self, radius: f64) -> Self {
        self.radius = radius.abs();
        self
    }

    // /// Set maximum radius
    // pub fn max_radius(mut self, max_radius: f64) -> Self {
    //     self.max_radius = max_radius.abs();
    //     self
    // }

    /// Set eta
    pub fn eta(mut self, eta: f64) -> Result<Self, Error> {
        if eta >= 10e-3 || eta <= 0.0 {
            return Err(ArgminError::InvalidParameter {
                text: "SR1TrustRegion: eta must be in (0, 10^-3).".to_string(),
            }
            .into());
        }
        self.eta = eta;
        Ok(self)
    }
}

impl<O, B, R> Solver<O> for SR1TrustRegion<B, R>
where
    O: ArgminOp<Output = f64, Hessian = B>,
    O::Param: Debug
        + Clone
        + Default
        + Serialize
        + ArgminSub<O::Param, O::Param>
        + ArgminAdd<O::Param, O::Param>
        + ArgminDot<O::Param, f64>
        + ArgminDot<O::Param, O::Hessian>
        + ArgminNorm<f64>
        + ArgminZeroLike
        + ArgminMul<f64, O::Param>,
    O::Hessian: Debug
        + Clone
        + Default
        + Serialize
        + DeserializeOwned
        + ArgminSub<O::Hessian, O::Hessian>
        + ArgminDot<O::Param, O::Param>
        + ArgminDot<O::Hessian, O::Hessian>
        + ArgminAdd<O::Hessian, O::Hessian>
        + ArgminMul<f64, O::Hessian>,
    R: ArgminTrustRegion + Solver<OpWrapper<O>>,
{
    const NAME: &'static str = "SR1 Trust Region";

    fn init(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &IterState<O>,
    ) -> Result<Option<ArgminIterData<O>>, Error> {
        let param = state.get_param();
        let cost = op.apply(&param)?;
        let grad = op.gradient(&param)?;
        let hessian = state
            .get_hessian()
            .unwrap_or_else(|| op.hessian(&param).unwrap());
        Ok(Some(
            ArgminIterData::new()
                .param(param)
                .cost(cost)
                .grad(grad)
                .hessian(hessian),
        ))
    }

    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &IterState<O>,
    ) -> Result<ArgminIterData<O>, Error> {
        let xk = state.get_param();
        let cost = state.get_cost();
        let prev_grad = state
            .get_grad()
            .unwrap_or_else(|| op.gradient(&xk).unwrap());
        let hessian: O::Hessian = state.get_hessian().unwrap();

        self.subproblem.set_radius(self.radius);

        let ArgminResult {
            operator: sub_op,
            state: IterState { param: sk, .. },
        } = Executor::new(
            OpWrapper::new_from_op(&op),
            self.subproblem.clone(),
            // xk.clone(),
            xk.zero_like(),
        )
        .cost(cost)
        .grad(prev_grad.clone())
        .hessian(hessian.clone())
        .ctrlc(false)
        .run()?;

        op.consume_op(sub_op);

        let xksk = xk.add(&sk);
        let dfk1 = op.gradient(&xksk)?;
        let yk = dfk1.sub(&prev_grad);
        let fk1 = op.apply(&xksk)?;

        let ared = cost - fk1;
        let tmp1: f64 = prev_grad.dot(&sk);
        let tmp2: f64 = sk.weighted_dot(&hessian, &sk);
        let tmp2: f64 = tmp2.mul(&0.5);
        let pred = -tmp1 - tmp2;
        let ap = ared / pred;

        let (xk1, fk1, dfk1) = if ap > self.eta {
            (xksk, fk1, dfk1)
        } else {
            (xk, cost, prev_grad)
        };

        self.radius = if ap > 0.75 {
            if sk.norm() <= 0.8 * self.radius {
                self.radius
            } else {
                2.0 * self.radius
            }
        } else if ap <= 0.75 && ap >= 0.1 {
            self.radius
        } else {
            0.5 * self.radius
        };

        let bksk = hessian.dot(&sk);
        let ykbksk = yk.sub(&bksk);
        let skykbksk: f64 = sk.dot(&ykbksk);

        let hessian_update = skykbksk.abs() >= self.r * sk.norm() * skykbksk.norm();
        let hessian = if hessian_update {
            let a: O::Hessian = ykbksk.dot(&ykbksk);
            let b: f64 = sk.dot(&ykbksk);
            hessian.add(&a.mul(&(1.0 / b)))
        } else {
            hessian
        };

        Ok(ArgminIterData::new()
            .param(xk1)
            .cost(fk1)
            .grad(dfk1)
            .hessian(hessian)
            .kv(make_kv!["ared" => ared;
                         "pred" => pred;
                         "ap" => ap;
                         "radius" => self.radius;
                         "hessian_update" => hessian_update;]))
    }

    fn terminate(&mut self, state: &IterState<O>) -> TerminationReason {
        /*std::f64::EPSILON.sqrt()*/
        if state.get_grad().unwrap().norm() < 1e-3 {
            return TerminationReason::TargetPrecisionReached;
        }
        // if (state.get_prev_cost() - state.get_cost()).abs() < std::f64::EPSILON {
        //     return TerminationReason::NoChangeInCost;
        // }
        TerminationReason::NotTerminated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_trait_impl;
    use crate::solver::trustregion::CauchyPoint;

    type Operator = MinimalNoOperator;

    test_trait_impl!(sr1, SR1TrustRegion<Operator, CauchyPoint>);
}
