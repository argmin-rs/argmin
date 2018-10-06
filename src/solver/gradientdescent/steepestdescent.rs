// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Argmin Steepest Descent

use prelude::*;
use solver::linesearch::HagerZhangLineSearch;
use std;
use std::default::Default;

// use solver::linesearch::BacktrackingLineSearch;

/// Template
#[derive(ArgminSolver)]
pub struct SteepestDescent<'a, T, H>
where
    T: 'a
        + Clone
        + Default
        + std::fmt::Debug
        + ArgminScale<f64>
        + ArgminSub<T>
        + ArgminNorm<f64>
        + ArgminDot<T, f64>
        + ArgminScaledAdd<T, f64>
        + ArgminScaledSub<T, f64>,
    H: 'a + Clone + Default,
{
    /// line search
    linesearch: Box<ArgminLineSearch<Parameters = T, OperatorOutput = f64, Hessian = H> + 'a>,
    /// Base stuff
    base: ArgminBase<T, f64, H>,
}

impl<'a, T, H> SteepestDescent<'a, T, H>
where
    T: 'a
        + Clone
        + Default
        + std::fmt::Debug
        + ArgminScale<f64>
        + ArgminSub<T>
        + ArgminNorm<f64>
        + ArgminDot<T, f64>
        + ArgminScaledAdd<T, f64>
        + ArgminScaledSub<T, f64>,
    H: 'a + Clone + Default,
{
    /// Constructor
    pub fn new(
        cost_function: Box<ArgminOperator<Parameters = T, OperatorOutput = f64, Hessian = H>>,
        init_param: T,
    ) -> Result<Self, Error> {
        let linesearch = HagerZhangLineSearch::new(cost_function.clone());
        Ok(SteepestDescent {
            linesearch: Box::new(linesearch),
            base: ArgminBase::new(cost_function, init_param),
        })
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

impl<'a, T, H> ArgminNextIter for SteepestDescent<'a, T, H>
where
    T: 'a
        + Clone
        + Default
        + std::fmt::Debug
        + ArgminScale<f64>
        + ArgminSub<T>
        + ArgminNorm<f64>
        + ArgminDot<T, f64>
        + ArgminScaledAdd<T, f64>
        + ArgminScaledSub<T, f64>,
    H: 'a + Clone + Default,
{
    type Parameters = T;
    type OperatorOutput = f64;
    type Hessian = H;

    /// Perform one iteration of SA algorithm
    fn next_iter(&mut self) -> Result<ArgminIterationData<Self::Parameters>, Error> {
        // reset line search
        self.linesearch.reset();

        let param_new = self.base.cur_param();
        let new_cost = self.apply(&param_new)?;
        let new_grad = self.gradient(&param_new)?;

        let norm = new_grad.norm();

        self.linesearch.set_initial_parameter(param_new);
        self.linesearch.set_initial_gradient(new_grad.clone());
        self.linesearch.set_initial_cost(new_cost);
        self.linesearch
            .set_search_direction(new_grad.scale(-1.0 / norm));

        self.linesearch.run_fast()?;

        let linesearch_result = self.linesearch.result();

        let out = ArgminIterationData::new(linesearch_result.param, linesearch_result.cost);
        Ok(out)
    }
}
