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

/// Newton's method iteratively finds the stationary points of a function f by using a second order
/// approximation of f at the current point.
///
/// [Example](https://github.com/argmin-rs/argmin/blob/master/examples/newton.rs)
///
/// # References:
///
/// [0] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(Clone, Serialize, Deserialize)]
pub struct Newton<F> {
    /// gamma
    gamma: F,
}

impl<F: ArgminFloat> Newton<F> {
    /// Constructor
    pub fn new() -> Self {
        Newton {
            gamma: F::from_f64(1.0).unwrap(),
        }
    }

    /// set gamma
    pub fn set_gamma(mut self, gamma: F) -> Result<Self, Error> {
        if gamma <= F::from_f64(0.0).unwrap() || gamma > F::from_f64(1.0).unwrap() {
            return Err(ArgminError::InvalidParameter {
                text: "Newton: gamma must be in  (0, 1].".to_string(),
            }
            .into());
        }
        self.gamma = gamma;
        Ok(self)
    }
}

impl<F: ArgminFloat> Default for Newton<F> {
    fn default() -> Newton<F> {
        Newton::new()
    }
}

impl<O, F> Solver<O> for Newton<F>
where
    O: ArgminOp<Float = F>,
    O::Param: ArgminScaledSub<O::Param, O::Float, O::Param>,
    O::Hessian: ArgminInv<O::Hessian> + ArgminDot<O::Param, O::Param>,
    F: ArgminFloat,
{
    const NAME: &'static str = "Newton method";

    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &IterState<O>,
    ) -> Result<ArgminIterData<O>, Error> {
        let param = state.get_param();
        let grad = op.gradient(&param)?;
        let hessian = op.hessian(&param)?;
        let new_param = param.scaled_sub(&self.gamma, &hessian.inv()?.dot(&grad));
        Ok(ArgminIterData::new().param(new_param))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_trait_impl;

    test_trait_impl!(newton_method, Newton<f64>);
}
