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
// use crate::solver::linesearch::HagerZhangLineSearch;
use crate::solver::linesearch::MoreThuenteLineSearch;
use std;
use std::default::Default;
use std::fmt::Debug;

/// Text
///
/// # Example
///
///
/// # References:
///
/// [0] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(ArgminSolver)]
#[stop("self.cur_grad().norm() < std::f64::EPSILON.sqrt()" => TargetPrecisionReached)]
pub struct BFGS<'a, T, H>
where
    T: 'a
        + Clone
        + Default
        + Debug
        + ArgminDot<T, f64>
        + ArgminDot<T, H>
        + ArgminNorm<f64>
        + ArgminScale<f64>
        + ArgminScaledAdd<T, f64>
        + ArgminScaledSub<T, f64>
        + ArgminSub<T>,
    H: 'a
        + Clone
        + Default
        + ArgminDot<T, T>
        + ArgminDot<H, H>
        + ArgminEye
        + ArgminSub<H>
        + ArgminAdd<H>
        + ArgminScale<f64>,
{
    /// Inverse Hessian
    inv_hessian: H,
    /// line search
    linesearch: Box<ArgminLineSearch<Parameters = T, OperatorOutput = f64, Hessian = H> + 'a>,
    /// Base stuff
    base: ArgminBase<'a, T, f64, H>,
}

impl<'a, T, H> BFGS<'a, T, H>
where
    T: 'a
        + Clone
        + Default
        + Debug
        + ArgminDot<T, f64>
        + ArgminDot<T, H>
        + ArgminNorm<f64>
        + ArgminScale<f64>
        + ArgminScaledAdd<T, f64>
        + ArgminScaledSub<T, f64>
        + ArgminSub<T>,
    H: 'a
        + Clone
        + Default
        + ArgminDot<T, T>
        + ArgminDot<H, H>
        + ArgminEye
        + ArgminSub<H>
        + ArgminAdd<H>
        + ArgminScale<f64>,
{
    /// Constructor
    pub fn new(
        cost_function: &'a ArgminOperator<Parameters = T, OperatorOutput = f64, Hessian = H>,
        init_param: T,
        init_inverse_hessian: H,
    ) -> Self {
        let linesearch = MoreThuenteLineSearch::new(cost_function);
        BFGS {
            inv_hessian: init_inverse_hessian,
            linesearch: Box::new(linesearch),
            base: ArgminBase::new(cost_function, init_param),
        }
    }

    /// Specify line search method
    pub fn set_linesearch(
        &mut self,
        linesearch: Box<ArgminLineSearch<Parameters = T, OperatorOutput = f64, Hessian = H> + 'a>,
    ) -> &mut Self {
        self.linesearch = linesearch;
        self
    }
}

impl<'a, T, H> ArgminNextIter for BFGS<'a, T, H>
where
    T: 'a
        + Clone
        + Default
        + Debug
        + ArgminDot<T, f64>
        + ArgminDot<T, H>
        + ArgminNorm<f64>
        + ArgminScale<f64>
        + ArgminScaledAdd<T, f64>
        + ArgminScaledSub<T, f64>
        + ArgminSub<T>,
    H: 'a
        + Clone
        + Default
        + ArgminDot<T, T>
        + ArgminDot<H, H>
        + ArgminEye
        + ArgminSub<H>
        + ArgminAdd<H>
        + ArgminScale<f64>,
{
    type Parameters = T;
    type OperatorOutput = f64;
    type Hessian = H;

    fn init(&mut self) -> Result<(), Error> {
        let grad = self.gradient(&self.base.cur_param())?;
        self.base.set_cur_grad(grad);
        Ok(())
    }

    fn next_iter(&mut self) -> Result<ArgminIterationData<Self::Parameters>, Error> {
        // reset line search
        self.linesearch.base_reset();

        let param = self.cur_param();
        let cur_cost = self.cur_cost();
        let prev_grad = self.base.cur_grad();
        let p = self.inv_hessian.dot(&prev_grad).scale(-1.0);

        self.linesearch.set_initial_parameter(param.clone());
        self.linesearch.set_initial_gradient(prev_grad.clone());
        self.linesearch.set_initial_cost(cur_cost);
        self.linesearch.set_search_direction(p);
        self.linesearch.run_fast()?;

        let linesearch_result = self.linesearch.result();
        let xk1 = linesearch_result.param;

        let grad = self.gradient(&xk1)?;
        let yk = grad.sub(&prev_grad);
        self.base.set_cur_grad(grad);

        let sk = xk1.sub(&param);

        let yksk: f64 = yk.dot(&sk);
        let rhok = 1.0 / yksk;

        let e = H::eye(2);
        let mat1: H = sk.dot(&yk);
        let mat1 = mat1.scale(rhok);
        // This is unnecessary ... however, there is no ArgminTranspose yet....
        let mat2: H = yk.dot(&sk);
        let mat2 = mat2.scale(rhok);

        let tmp1 = e.sub(&mat1);
        let tmp2 = e.sub(&mat2);

        // TODO: Update H
        let sksk: H = sk.dot(&sk);
        let sksk = sksk.scale(rhok);

        self.inv_hessian = tmp1.dot(&self.inv_hessian.dot(&tmp2)).add(&sksk);

        let out = ArgminIterationData::new(xk1, linesearch_result.cost);
        Ok(out)
    }
}
