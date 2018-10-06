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
pub struct CauchyPoint<T, H>
where
    T: Clone
        + std::default::Default
        + std::fmt::Debug
        + ArgminWeightedDot<T, f64, H>
        + ArgminNorm<f64>
        + ArgminScale<f64>,
    H: Clone + std::default::Default,
{
    /// Radius
    radius: f64,
    /// base
    base: ArgminBase<T, f64, H>,
}

impl<T, H> CauchyPoint<T, H>
where
    T: Clone
        + std::default::Default
        + std::fmt::Debug
        + ArgminWeightedDot<T, f64, H>
        + ArgminNorm<f64>
        + ArgminScale<f64>,
    H: Clone + std::default::Default,
{
    /// Constructor
    ///
    /// Parameters:
    ///
    /// `operator`: operator
    pub fn new(
        operator: Box<ArgminOperator<Parameters = T, OperatorOutput = f64, Hessian = H>>,
    ) -> Self {
        let base = ArgminBase::new(operator, T::default());
        CauchyPoint {
            radius: std::f64::NAN,
            base: base,
        }
    }
}

impl<T, H> ArgminNextIter for CauchyPoint<T, H>
where
    T: Clone
        + std::default::Default
        + std::fmt::Debug
        + ArgminWeightedDot<T, f64, H>
        + ArgminNorm<f64>
        + ArgminScale<f64>,
    H: Clone + std::default::Default,
{
    type Parameters = T;
    type OperatorOutput = f64;
    type Hessian = H;

    fn init(&mut self) -> Result<(), Error> {
        self.base.reset();
        // This is not an iterative method.
        self.set_max_iters(1);
        Ok(())
    }

    fn next_iter(&mut self) -> Result<ArgminIterationData<Self::Parameters>, Error> {
        let grad = self.base.cur_grad();
        let grad_norm = grad.norm();
        let wdp = grad.weighted_dot(self.base.cur_hessian().clone(), grad.clone());
        let tau: f64 = if wdp <= 0.0 {
            1.0
        } else {
            1.0f64.min(grad_norm.powi(3) / (self.radius * wdp))
        };

        let new_param = grad.scale(-tau * self.radius / grad_norm);
        let out = ArgminIterationData::new(new_param, 0.0);
        Ok(out)
    }
}

impl<T, H> ArgminTrustRegion for CauchyPoint<T, H>
where
    T: Clone
        + std::default::Default
        + std::fmt::Debug
        + ArgminWeightedDot<T, f64, H>
        + ArgminNorm<f64>
        + ArgminScale<f64>,
    H: Clone + std::default::Default,
{
    // fn set_initial_parameter(&mut self, param: T) {
    //     self.base.set_cur_param(param);
    // }

    fn set_radius(&mut self, radius: f64) {
        self.radius = radius;
    }

    fn set_grad(&mut self, grad: T) {
        self.base.set_cur_grad(grad);
    }

    fn set_hessian(&mut self, hessian: H) {
        self.base.set_cur_hessian(hessian);
    }
}
