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
use serde::{Deserialize, Serialize};
use std::default::Default;

/// Gauss-Newton method
///
/// [Example](https://github.com/argmin-rs/argmin/blob/master/examples/gaussnewton.rs)
///
/// # References:
///
/// [0] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(Clone, Serialize, Deserialize)]
pub struct GaussNewton<F> {
    /// gamma
    gamma: F,
    /// Tolerance for the stopping criterion based on cost difference
    tol: F,
}

impl<F: ArgminFloat> GaussNewton<F> {
    /// Constructor
    pub fn new() -> Self {
        GaussNewton {
            gamma: F::from_f64(1.0).unwrap(),
            tol: F::epsilon().sqrt(),
        }
    }

    /// set gamma
    pub fn with_gamma(mut self, gamma: F) -> Result<Self, Error> {
        if gamma <= F::from_f64(0.0).unwrap() || gamma > F::from_f64(1.0).unwrap() {
            return Err(ArgminError::InvalidParameter {
                text: "Gauss-Newton: gamma must be in  (0, 1].".to_string(),
            }
            .into());
        }
        self.gamma = gamma;
        Ok(self)
    }

    /// Set tolerance for the stopping criterion based on cost difference
    pub fn with_tol(mut self, tol: F) -> Result<Self, Error> {
        if tol <= F::from_f64(0.0).unwrap() {
            return Err(ArgminError::InvalidParameter {
                text: "Gauss-Newton: tol must be positive.".to_string(),
            }
            .into());
        }
        self.tol = tol;
        Ok(self)
    }
}

impl<F: ArgminFloat> Default for GaussNewton<F> {
    fn default() -> GaussNewton<F> {
        GaussNewton::new()
    }
}

impl<O, F> Solver<O> for GaussNewton<F>
where
    O: ArgminOp<Float = F>,
    O::Param: Default
        + ArgminScaledSub<O::Param, O::Float, O::Param>
        + ArgminSub<O::Param, O::Param>
        + ArgminMul<O::Float, O::Param>,
    O::Output: ArgminNorm<O::Float>,
    O::Jacobian: ArgminTranspose
        + ArgminInv<O::Jacobian>
        + ArgminDot<O::Jacobian, O::Jacobian>
        + ArgminDot<O::Output, O::Param>
        + ArgminDot<O::Param, O::Param>,
    O::Hessian: Default,
    F: ArgminFloat,
{
    const NAME: &'static str = "Gauss-Newton method";

    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &IterState<O>,
    ) -> Result<ArgminIterData<O>, Error> {
        let param = state.get_param();
        let residuals = op.apply(&param)?;
        let jacobian = op.jacobian(&param)?;

        let p = jacobian
            .clone()
            .t()
            .dot(&jacobian)
            .inv()?
            .dot(&jacobian.t().dot(&residuals));

        let new_param = param.sub(&p.mul(&self.gamma));

        Ok(ArgminIterData::new()
            .param(new_param)
            .cost(residuals.norm()))
    }

    fn terminate(&mut self, state: &IterState<O>) -> TerminationReason {
        if (state.get_prev_cost() - state.get_cost()).abs() < self.tol {
            return TerminationReason::NoChangeInCost;
        }
        TerminationReason::NotTerminated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_trait_impl;

    test_trait_impl!(gauss_newton_method, GaussNewton<f64>);

    #[test]
    fn test_tolerance() {
        let tol1: f64 = 1e-4;

        let GaussNewton { tol: t, .. } = GaussNewton::new().with_tol(tol1).unwrap();

        assert!((t - tol1).abs() < std::f64::EPSILON);
    }

    #[test]
    fn test_gamma() {
        let gamma: f64 = 0.5;

        let GaussNewton { gamma: g, .. } = GaussNewton::new().with_gamma(gamma).unwrap();

        assert!((g - gamma).abs() < std::f64::EPSILON);
    }
}
