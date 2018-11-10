// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Newton-CG method
//!
//! TODO: Stop when search direction is close to 0
//!
//! # References:
//!
//! [0] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
//! Springer. ISBN 0-387-30303-0.

use crate::solver::conjugategradient::ConjugateGradient;
// use crate::solver::linesearch::HagerZhangLineSearch;
use crate::solver::linesearch::MoreThuenteLineSearch;
use prelude::*;
use std;
use std::default::Default;
use std::fmt::Debug;

/// Newton-CG Method
#[derive(ArgminSolver)]
pub struct NewtonCG<'a, T, H>
where
    T: 'a
        + Clone
        + Default
        + Debug
        + ArgminScaledSub<T, f64>
        + ArgminScaledAdd<T, f64>
        + ArgminDot<T, f64>
        + ArgminAdd<T>
        + ArgminSub<T>
        + ArgminZero
        + ArgminNorm<f64>
        + ArgminScale<f64>,
    H: 'a + Clone + Default + ArgminInv<H> + ArgminDot<T, T>,
{
    /// line search
    linesearch: Box<ArgminLineSearch<Parameters = T, OperatorOutput = f64, Hessian = H> + 'a>,
    /// Base stuff
    base: ArgminBase<'a, T, f64, H>,
}

impl<'a, T, H> NewtonCG<'a, T, H>
where
    T: 'a
        + Clone
        + Default
        + Debug
        + ArgminScaledSub<T, f64>
        + ArgminScaledAdd<T, f64>
        + ArgminDot<T, f64>
        + ArgminAdd<T>
        + ArgminSub<T>
        + ArgminZero
        + ArgminNorm<f64>
        + ArgminScale<f64>,
    H: 'a + Clone + Default + ArgminInv<H> + ArgminDot<T, T>,
{
    /// Constructor
    pub fn new(
        cost_function: &'a ArgminOperator<Parameters = T, OperatorOutput = f64, Hessian = H>,
        init_param: T,
    ) -> Self {
        // let linesearch = HagerZhangLineSearch::new(cost_function);
        let linesearch = MoreThuenteLineSearch::new(cost_function);
        NewtonCG {
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

impl<'a, T, H> ArgminNextIter for NewtonCG<'a, T, H>
where
    T: 'a
        + Clone
        + Default
        + Debug
        + ArgminScaledSub<T, f64>
        + ArgminScaledAdd<T, f64>
        + ArgminDot<T, f64>
        + ArgminAdd<T>
        + ArgminSub<T>
        + ArgminZero
        + ArgminNorm<f64>
        + ArgminScale<f64>,
    H: 'a + Clone + Default + ArgminInv<H> + ArgminDot<T, T>,
{
    type Parameters = T;
    type OperatorOutput = f64;
    type Hessian = H;

    fn next_iter(&mut self) -> Result<ArgminIterationData<Self::Parameters>, Error> {
        let param = self.cur_param();
        let grad = self.gradient(&param)?;
        let hessian = self.hessian(&param)?;

        // Solve CG subproblem
        let op: CGSubProblem<'a, T, H> = CGSubProblem::new(hessian.clone());

        let mut x_p = param.zero();
        let mut x: T = param.zero();
        let mut cg = ConjugateGradient::new(&op, grad.scale(-1.0), x_p.clone())?;

        cg.init()?;
        let grad_norm = grad.norm();
        for iter in 0.. {
            let data = cg.next_iter()?;
            x = data.param();
            cg.increment_iter();
            cg.set_cur_param(data.param());
            cg.set_cur_cost(data.cost());
            let p = cg.p_prev();
            // let p = cg.p();
            let curvature = p.dot(hessian.dot(p.clone()));
            // println!("iter: {:?}, curv: {:?}", iter, curvature);
            if curvature <= 0.0 {
                if iter == 0 {
                    x = grad.scale(-1.0);
                    break;
                } else {
                    x = x_p;
                    break;
                }
            }
            if data.cost() <= (0.5f64).min(grad_norm.sqrt()) * grad_norm {
                break;
            }
            x_p = x.clone();
        }

        // perform line search
        self.linesearch.base_reset();
        self.linesearch.set_initial_parameter(param);
        self.linesearch.set_initial_gradient(grad);
        let cost = self.cur_cost();
        self.linesearch.set_initial_cost(cost);
        self.linesearch.set_search_direction(x);

        self.linesearch.run_fast()?;

        let linesearch_result = self.linesearch.result();

        // take care of counting
        let cost_count_cg = cg.cost_func_count();
        let grad_count_cg = cg.grad_func_count();
        let hessian_count_cg = cg.hessian_func_count();
        let cost_count_ls = self.linesearch.cost_func_count();
        let grad_count_ls = self.linesearch.grad_func_count();
        let hessian_count_ls = self.linesearch.hessian_func_count();
        self.increase_cost_func_count(cost_count_cg + cost_count_ls);
        self.increase_grad_func_count(grad_count_cg + grad_count_ls);
        self.increase_hessian_func_count(hessian_count_cg + hessian_count_ls);

        let out = ArgminIterationData::new(linesearch_result.param, linesearch_result.cost);
        Ok(out)
    }
}

#[derive(Clone)]
struct CGSubProblem<'a, T, H>
where
    H: 'a + Clone + Default + ArgminDot<T, T>,
    T: 'a + Clone,
{
    hessian: H,
    phantom: std::marker::PhantomData<&'a T>,
}

impl<'a, T, H> CGSubProblem<'a, T, H>
where
    H: 'a + Clone + Default + ArgminDot<T, T>,
    T: 'a + Clone,
{
    /// constructor
    pub fn new(hessian: H) -> Self {
        CGSubProblem {
            hessian,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<'a, T, H> ArgminOperator for CGSubProblem<'a, T, H>
where
    T: 'a + Clone,
    H: 'a + Clone + Default + ArgminDot<T, T>,
{
    type Parameters = T;
    type OperatorOutput = T;
    type Hessian = ();

    fn apply(&self, p: &T) -> Result<T, Error> {
        Ok(self.hessian.dot(p.clone()))
    }

    /// dont ever clone this
    fn box_clone(&self) -> Box<ArgminOperator<Parameters = T, OperatorOutput = T, Hessian = ()>> {
        unimplemented!()
    }
}
