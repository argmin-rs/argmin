// Copyright 2019-2022 argmin developers
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
    Hessian, IterState, OpWrapper, OptimizationResult, SerializeAlias, Solver, TerminationReason,
    TrustRegionRadius, KV,
};
use argmin_math::{
    ArgminAdd, ArgminDot, ArgminMul, ArgminNorm, ArgminSub, ArgminWeightedDot, ArgminZeroLike,
};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// SR1 Trust Region method
///
/// # References:
///
/// \[0\] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
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

impl<B, R, F> SR1TrustRegion<B, R, F>
where
    F: ArgminFloat,
{
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

    /// provide initial Hessian (it will be computed if not provided)
    #[must_use]
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
    #[must_use]
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
    #[must_use]
    pub fn with_tol_grad(mut self, tol_grad: F) -> Self {
        self.tol_grad = tol_grad;
        self
    }
}

impl<O, R, P, G, B, F> Solver<O, IterState<P, G, (), B, F>> for SR1TrustRegion<B, R, F>
where
    O: CostFunction<Param = P, Output = F>
        + Gradient<Param = P, Gradient = G>
        + Hessian<Param = P, Hessian = B>,
    P: Clone
        + SerializeAlias
        + DeserializeOwnedAlias
        + ArgminSub<P, P>
        + ArgminAdd<P, P>
        + ArgminDot<P, F>
        + ArgminDot<P, B>
        + ArgminNorm<F>
        + ArgminZeroLike,
    G: Clone
        + SerializeAlias
        + DeserializeOwnedAlias
        + ArgminNorm<F>
        + ArgminDot<P, F>
        + ArgminSub<G, P>,
    B: Clone
        + SerializeAlias
        + DeserializeOwnedAlias
        + ArgminDot<P, P>
        + ArgminAdd<B, B>
        + ArgminMul<F, B>,
    R: Clone + TrustRegionRadius<F> + Solver<O, IterState<P, G, (), B, F>>,
    F: ArgminFloat + ArgminNorm<F>,
{
    const NAME: &'static str = "SR1 Trust Region";

    fn init(
        &mut self,
        op: &mut OpWrapper<O>,
        mut state: IterState<P, G, (), B, F>,
    ) -> Result<(IterState<P, G, (), B, F>, Option<KV>), Error> {
        let param = state.take_param().unwrap();
        let cost = op.cost(&param)?;
        let grad = op.gradient(&param)?;
        let hessian = state
            .take_hessian()
            .map(Result::Ok)
            .unwrap_or_else(|| op.hessian(&param))?;
        Ok((
            state.param(param).cost(cost).grad(grad).hessian(hessian),
            None,
        ))
    }

    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        mut state: IterState<P, G, (), B, F>,
    ) -> Result<(IterState<P, G, (), B, F>, Option<KV>), Error> {
        let xk = state.take_param().unwrap();
        let cost = state.cost;
        let prev_grad = state
            .take_grad()
            .map(Result::Ok)
            .unwrap_or_else(|| op.gradient(&xk))?;
        let hessian = state.take_hessian().unwrap();

        self.subproblem.set_radius(self.radius);

        let OptimizationResult {
            operator: sub_op,
            state: mut sub_state,
        } = Executor::new(op.take_op().unwrap(), self.subproblem.clone())
            .configure(|config| {
                config
                    .param(xk.zero_like())
                    .hessian(hessian.clone())
                    .grad(prev_grad.clone())
                    .cost(cost)
            })
            .ctrlc(false)
            .run()?;

        let sk = sub_state.take_param().unwrap();

        op.consume_op(sub_op);

        let xksk = xk.add(&sk);
        let dfk1 = op.gradient(&xksk)?;
        let yk = dfk1.sub(&prev_grad);
        let fk1 = op.cost(&xksk)?;

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
            let a: B = ykbksk.dot(&ykbksk);
            let b: F = sk.dot(&ykbksk);
            hessian.add(&a.mul(&(F::from_f64(1.0).unwrap() / b)))
        } else {
            hessian
        };

        Ok((
            state.param(xk1).cost(fk1).grad(dfk1).hessian(hessian),
            Some(make_kv!["ared" => ared;
                         "pred" => pred;
                         "ap" => ap;
                         "radius" => self.radius;
                         "hessian_update" => hessian_update;]),
        ))
    }

    fn terminate(&mut self, state: &IterState<P, G, (), B, F>) -> TerminationReason {
        if state.get_grad_ref().unwrap().norm() < self.tol_grad {
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
    use crate::core::PseudoOperator;
    use crate::solver::trustregion::CauchyPoint;
    use crate::test_trait_impl;

    type Operator = PseudoOperator;

    test_trait_impl!(sr1, SR1TrustRegion<Operator, CauchyPoint<f64>, f64>);

    #[test]
    fn test_tolerances() {
        let subproblem = CauchyPoint::new();

        let tol: f64 = 1e-4;

        let SR1TrustRegion { tol_grad: t, .. }: SR1TrustRegion<
            PseudoOperator,
            CauchyPoint<f64>,
            f64,
        > = SR1TrustRegion::new(subproblem).with_tol_grad(tol);

        assert!((t - tol).abs() < std::f64::EPSILON);
    }
}
