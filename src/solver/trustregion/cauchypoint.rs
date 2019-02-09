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
use std;

/// The Cauchy point is the minimum of the quadratic approximation of the cost function within the
/// trust region along the direction given by the first derivative.
///
/// # References:
///
/// [0] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(ArgminSolver)]
pub struct CauchyPoint<T, H, O>
where
    T: Clone
        + std::default::Default
        + std::fmt::Debug
        + ArgminWeightedDot<T, f64, H>
        + ArgminNorm<f64>
        + ArgminMul<f64, T>,
    H: Clone + std::default::Default,
    O: ArgminOperator<Parameters = T, OperatorOutput = f64, Hessian = H>,
{
    /// Radius
    radius: f64,
    /// base
    base: ArgminBase<T, H, O>,
}

impl<T, H, O> CauchyPoint<T, H, O>
where
    T: Clone
        + std::default::Default
        + std::fmt::Debug
        + ArgminWeightedDot<T, f64, H>
        + ArgminNorm<f64>
        + ArgminMul<f64, T>,
    H: Clone + std::default::Default,
    O: ArgminOperator<Parameters = T, OperatorOutput = f64, Hessian = H>,
{
    /// Constructor
    ///
    /// Parameters:
    ///
    /// `operator`: operator
    pub fn new(operator: O) -> Self {
        let base = ArgminBase::new(operator, T::default());
        CauchyPoint {
            radius: std::f64::NAN,
            base,
        }
    }
}

impl<T, H, O> ArgminNextIter for CauchyPoint<T, H, O>
where
    T: Clone
        + std::default::Default
        + std::fmt::Debug
        + ArgminWeightedDot<T, f64, H>
        + ArgminNorm<f64>
        + ArgminMul<f64, T>,
    H: Clone + std::default::Default,
    O: ArgminOperator<Parameters = T, OperatorOutput = f64, Hessian = H>,
{
    type Parameters = T;
    type OperatorOutput = f64;
    type Hessian = H;

    fn init(&mut self) -> Result<(), Error> {
        self.base_reset();
        // This is not an iterative method.
        self.set_max_iters(1);
        Ok(())
    }

    fn next_iter(&mut self) -> Result<ArgminIterationData<Self::Parameters>, Error> {
        let grad = self.cur_grad();
        let grad_norm = grad.norm();
        let wdp = grad.weighted_dot(&self.cur_hessian(), &grad);
        let tau: f64 = if wdp <= 0.0 {
            1.0
        } else {
            1.0f64.min(grad_norm.powi(3) / (self.radius * wdp))
        };

        let new_param = grad.mul(&(-tau * self.radius / grad_norm));
        let out = ArgminIterationData::new(new_param, 0.0);
        Ok(out)
    }
}

impl<T, H, O> ArgminTrustRegion for CauchyPoint<T, H, O>
where
    T: Clone
        + std::default::Default
        + std::fmt::Debug
        + ArgminWeightedDot<T, f64, H>
        + ArgminNorm<f64>
        + ArgminMul<f64, T>,
    H: Clone + std::default::Default,
    O: ArgminOperator<Parameters = T, OperatorOutput = f64, Hessian = H>,
{
    fn set_radius(&mut self, radius: f64) {
        self.radius = radius;
    }

    fn set_grad(&mut self, grad: T) {
        self.set_cur_grad(grad);
    }

    fn set_hessian(&mut self, hessian: H) {
        self.set_cur_hessian(hessian);
    }
}
