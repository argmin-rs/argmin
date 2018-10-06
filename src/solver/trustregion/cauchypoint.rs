// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Cauchy point
//!
//!
//! ## Reference
//!
//! TODO
//!
// //!
// //! # Example
// //!
// //! ```rust
// //! todo
// //! ```

use prelude::*;
use std;

/// Cauchy Point
#[derive(ArgminSolver)]
pub struct CauchyPoint<T, U>
where
    T: Clone
        + std::default::Default
        + std::fmt::Debug
        + ArgminWeightedDot<T, f64, U>
        + ArgminNorm<f64>
        + ArgminScaledAdd<T, f64>,
    U: Clone,
{
    /// Radius
    radius: f64,
    /// Hessian
    b: U,
    /// base
    base: ArgminBase<T, f64>,
}

impl<T, U> CauchyPoint<T, U>
where
    T: Clone
        + std::default::Default
        + std::fmt::Debug
        + ArgminWeightedDot<T, f64, U>
        + ArgminNorm<f64>
        + ArgminScaledAdd<T, f64>,
    U: Clone,
{
    /// Constructor
    ///
    /// Parameters:
    ///
    /// `operator`: operator
    pub fn new(
        operator: Box<ArgminOperator<Parameters = T, OperatorOutput = f64>>,
        radius: f64,
        param: T,
        grad: T,
        hessian: U,
    ) -> Self {
        let mut base = ArgminBase::new(operator, T::default());
        base.set_cur_param(param);
        base.set_cur_grad(grad);
        CauchyPoint {
            radius: radius,
            b: hessian,
            base: base,
        }
    }
}

impl<T, U> ArgminNextIter for CauchyPoint<T, U>
where
    T: Clone
        + std::default::Default
        + std::fmt::Debug
        + ArgminWeightedDot<T, f64, U>
        + ArgminNorm<f64>
        + ArgminScaledAdd<T, f64>,
    U: Clone,
{
    type Parameters = T;
    type OperatorOutput = f64;

    fn init(&mut self) -> Result<(), Error> {
        // This is not an iterative method.
        self.set_max_iters(1);
        Ok(())
    }

    fn next_iter(&mut self) -> Result<ArgminIterationData<Self::Parameters>, Error> {
        let grad = self.base.cur_grad();
        let grad_norm = grad.norm();
        let wdp = grad.weighted_dot(self.b.clone(), grad.clone());
        let tau: f64 = if wdp <= 0.0 {
            1.0
        } else {
            1.0f64.min(grad_norm.powi(3) / (self.radius * wdp))
        };

        let new_param = self
            .base
            .cur_param()
            .scaled_add(-tau * self.radius / grad_norm, grad.clone());
        let out = ArgminIterationData::new(new_param, std::f64::NAN);
        Ok(out)
    }
}
