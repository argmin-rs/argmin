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

/// The Cauchy point is the minimum of the quadratic approximation of the cost function within the
/// trust region along the direction given by the first derivative.
///
/// # References:
///
/// [0] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(ArgminSolver)]
pub struct CauchyPoint<O>
where
    O: ArgminOp<Output = f64>,
    <O as ArgminOp>::Param:
        ArgminMul<f64, <O as ArgminOp>::Param>
            + ArgminWeightedDot<<O as ArgminOp>::Param, f64, <O as ArgminOp>::Hessian>
            + ArgminNorm<f64>,
{
    /// Radius
    radius: f64,
    /// base
    base: ArgminBase<O>,
}

impl<O> CauchyPoint<O>
where
    O: ArgminOp<Output = f64>,
    <O as ArgminOp>::Param:
        ArgminMul<f64, <O as ArgminOp>::Param>
            + ArgminWeightedDot<<O as ArgminOp>::Param, f64, <O as ArgminOp>::Hessian>
            + ArgminNorm<f64>,
{
    /// Constructor
    ///
    /// Parameters:
    ///
    /// `operator`: operator
    pub fn new(operator: O) -> Self {
        let base = ArgminBase::new(operator, <O as ArgminOp>::Param::default());
        CauchyPoint {
            radius: std::f64::NAN,
            base,
        }
    }
}

impl<O> ArgminIter for CauchyPoint<O>
where
    O: ArgminOp<Output = f64>,
    <O as ArgminOp>::Param:
        ArgminMul<f64, <O as ArgminOp>::Param>
            + ArgminWeightedDot<<O as ArgminOp>::Param, f64, <O as ArgminOp>::Hessian>
            + ArgminNorm<f64>,
{
    type Param = <O as ArgminOp>::Param;
    type Output = <O as ArgminOp>::Output;
    type Hessian = <O as ArgminOp>::Hessian;

    fn init(&mut self) -> Result<(), Error> {
        self.base_reset();
        // This is not an iterative method.
        self.set_max_iters(1);
        Ok(())
    }

    fn next_iter(&mut self) -> Result<ArgminIterData<Self::Param>, Error> {
        let grad = self.cur_grad();
        let grad_norm = grad.norm();
        let wdp = grad.weighted_dot(&self.cur_hessian(), &grad);
        let tau: f64 = if wdp <= 0.0 {
            1.0
        } else {
            1.0f64.min(grad_norm.powi(3) / (self.radius * wdp))
        };

        let new_param = grad.mul(&(-tau * self.radius / grad_norm));
        let out = ArgminIterData::new(new_param, 0.0);
        Ok(out)
    }
}

impl<O> ArgminTrustRegion for CauchyPoint<O>
where
    O: ArgminOp<Output = f64>,
    <O as ArgminOp>::Param:
        ArgminMul<f64, <O as ArgminOp>::Param>
            + ArgminWeightedDot<<O as ArgminOp>::Param, f64, <O as ArgminOp>::Hessian>
            + ArgminNorm<f64>,
{
    fn set_radius(&mut self, radius: f64) {
        self.radius = radius;
    }

    fn set_grad(&mut self, grad: <O as ArgminOp>::Param) {
        self.set_cur_grad(grad);
    }

    fn set_hessian(&mut self, hessian: <O as ArgminOp>::Hessian) {
        self.set_cur_hessian(hessian);
    }
}