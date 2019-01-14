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
use crate::solver::linesearch::HagerZhangLineSearch;
use std;
use std::default::Default;

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
pub struct BFGS<'a, T, H>
where
    T: 'a
        + Clone
        + Default
        + ArgminDot<T, f64>
        + ArgminScale<f64>
        + ArgminScaledAdd<T, f64>
        + ArgminScaledSub<T, f64>
        + ArgminSub<T>,
    H: 'a + Clone + Default + ArgminDot<T, T>,
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
        + ArgminDot<T, f64>
        + ArgminScale<f64>
        + ArgminScaledAdd<T, f64>
        + ArgminScaledSub<T, f64>
        + ArgminSub<T>,
    H: 'a + Clone + Default + ArgminDot<T, T>,
{
    /// Constructor
    pub fn new(
        cost_function: &'a ArgminOperator<Parameters = T, OperatorOutput = f64, Hessian = H>,
        init_param: T,
        init_inverse_hessian: H,
    ) -> Self {
        let linesearch = HagerZhangLineSearch::new(cost_function);
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
        + ArgminDot<T, f64>
        + ArgminScale<f64>
        + ArgminScaledAdd<T, f64>
        + ArgminScaledSub<T, f64>
        + ArgminSub<T>,
    H: 'a + Clone + Default + ArgminDot<T, T>,
{
    type Parameters = T;
    type OperatorOutput = f64;
    type Hessian = H;

    fn next_iter(&mut self) -> Result<ArgminIterationData<Self::Parameters>, Error> {
        // reset line search
        self.linesearch.base_reset();

        let param = self.cur_param();
        let cur_cost = self.cur_cost();
        let grad = self.gradient(&param)?;
        let p = self.inv_hessian.dot(&grad).scale(-1.0);

        self.linesearch.set_initial_parameter(param);
        self.linesearch.set_initial_gradient(grad);
        self.linesearch.set_initial_cost(cur_cost);
        self.linesearch.set_search_direction(p);

        self.linesearch.run_fast()?;

        let linesearch_result = self.linesearch.result();

        // TODO: Update H

        let out = ArgminIterationData::new(linesearch_result.param, linesearch_result.cost);
        Ok(out)
    }
}
