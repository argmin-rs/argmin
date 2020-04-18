// Copyright 2019-2020 argmin developers
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
pub struct SR1TrustRegion<B, R, F> {
    /// parameter for skipping rule
    r: F,
    /// Inverse Hessian
    init_hessian: Option<B>,
    /// subproblem
    subproblem: R,
    /// Radius
    radius: F,
    /// eta \in [0, 1/4)
    eta: F,
    /// Tolerance for the stopping criterion based on the change of the norm on the gradient
    tol_grad: F,
}

impl<B, R, F: ArgminFloat> SR1TrustRegion<B, R, F> {
    /// Constructor
    pub fn new(subproblem: R) -> Self {
        SR1TrustRegion {
            r: F::from_f64(1e-8).unwrap(),
            init_hessian: None,
            subproblem,
            radius: F::from_f64(1.0).unwrap(),
            eta: F::from_f64(0.5 * 1e-3).unwrap(),
            tol_grad: F::from_f64(1e-3).unwrap(),
        }
    }

    /// provide initial Hessian (if not provided, the algorithm will try to compute it using the
    /// `hessian` method of `ArgminOp`.
    pub fn hessian(mut self, init_hessian: B) -> Self {
        self.init_hessian = Some(init_hessian);
        self
    }

    /// Set r
    pub fn r(mut self, r: F) -> Result<Self, Error> {
        if r <= F::from_f64(0.0).unwrap() || r >= F::from_f64(1.0).unwrap() {
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
    pub fn radius(mut self, radius: F) -> Self {
        self.radius = radius.abs();
        self
    }

    /// Set eta
    pub fn eta(mut self, eta: F) -> Result<Self, Error> {
        if eta >= F::from_f64(10e-3).unwrap() || eta <= F::from_f64(0.0).unwrap() {
            return Err(ArgminError::InvalidParameter {
                text: "SR1TrustRegion: eta must be in (0, 10^-3).".to_string(),
            }
            .into());
        }
        self.eta = eta;
        Ok(self)
    }

    /// Sets tolerance for the stopping criterion based on the change of the norm on the gradient
    pub fn with_tol_grad(mut self, tol_grad: F) -> Self {
        self.tol_grad = tol_grad;
        self
    }
}

impl<O, B, R, F> Solver<O, F> for SR1TrustRegion<B, R, F>
where
    O: ArgminOp<Output = F, Hessian = B>,
    O::Param: Debug
        + Clone
        + Default
        + Serialize
        + ArgminSub<O::Param, O::Param>
        + ArgminAdd<O::Param, O::Param>
        + ArgminDot<O::Param, F>
        + ArgminDot<O::Param, O::Hessian>
        + ArgminNorm<F>
        + ArgminZeroLike
        + ArgminMul<F, O::Param>,
    O::Hessian: Debug
        + Clone
        + Default
        + Serialize
        + DeserializeOwned
        + ArgminSub<O::Hessian, O::Hessian>
        + ArgminDot<O::Param, O::Param>
        + ArgminDot<O::Hessian, O::Hessian>
        + ArgminAdd<O::Hessian, O::Hessian>
        + ArgminMul<F, O::Hessian>,
    R: ArgminTrustRegion<F> + Solver<OpWrapper<O>, F>,
    F: ArgminFloat + ArgminNorm<F>,
{
    const NAME: &'static str = "SR1 Trust Region";

    fn init(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &IterState<O, F>,
    ) -> Result<Option<ArgminIterData<O, F>>, Error> {
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
        state: &IterState<O, F>,
    ) -> Result<ArgminIterData<O, F>, Error> {
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
            OpWrapper::new_from_wrapper(op),
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
        let tmp1: F = prev_grad.dot(&sk);
        let tmp2: F = sk.weighted_dot(&hessian, &sk);
        let tmp2: F = tmp2.mul(F::from_f64(0.5).unwrap());
        let pred = -tmp1 - tmp2;
        let ap = ared / pred;

        let (xk1, fk1, dfk1) = if ap > self.eta {
            (xksk, fk1, dfk1)
        } else {
            (xk, cost, prev_grad)
        };

        self.radius = if ap > F::from_f64(0.75).unwrap() {
            if sk.norm() <= F::from_f64(0.8).unwrap() * self.radius {
                self.radius
            } else {
                F::from_f64(2.0).unwrap() * self.radius
            }
        } else if ap <= F::from_f64(0.75).unwrap() && ap >= F::from_f64(0.1).unwrap() {
            self.radius
        } else {
            F::from_f64(0.5).unwrap() * self.radius
        };

        let bksk = hessian.dot(&sk);
        let ykbksk = yk.sub(&bksk);
        let skykbksk: F = sk.dot(&ykbksk);

        let hessian_update = skykbksk.abs() >= self.r * sk.norm() * skykbksk.norm();
        let hessian = if hessian_update {
            let a: O::Hessian = ykbksk.dot(&ykbksk);
            let b: F = sk.dot(&ykbksk);
            hessian.add(&a.mul(&(F::from_f64(1.0).unwrap() / b)))
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

    fn terminate(&mut self, state: &IterState<O, F>) -> TerminationReason {
        if state.get_grad().unwrap().norm() < self.tol_grad {
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
    use crate::solver::trustregion::CauchyPoint;
    use crate::test_trait_impl;

    type Operator = MinimalNoOperator;

    test_trait_impl!(sr1, SR1TrustRegion<Operator, CauchyPoint<f64>, f64>);

    #[test]
    fn test_tolerances() {
        let subproblem = CauchyPoint::new();

        let tol: f64 = 1e-4;

        let SR1TrustRegion { tol_grad: t, .. }: SR1TrustRegion<
            MinimalNoOperator,
            CauchyPoint<f64>,
            f64,
        > = SR1TrustRegion::new(subproblem).with_tol_grad(tol);

        assert!((t - tol).abs() < std::f64::EPSILON);
    }
}
