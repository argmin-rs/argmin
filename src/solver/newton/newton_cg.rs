// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! TODO: Stop when search direction is close to 0
//!
//! # References:
//!
//! [0] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
//! Springer. ISBN 0-387-30303-0.

use crate::prelude::*;
use crate::solver::conjugategradient::ConjugateGradient;
use crate::solver::linesearch::MoreThuenteLineSearch;
// use crate::solver::linesearch::HagerZhangLineSearch;
use std;
use std::default::Default;
use std::fmt::Debug;

/// The Newton-CG method (also called truncated Newton method) uses a modified CG to solve the
/// Newton equations approximately. After a search direction is found, a line search is performed.
///
/// # Example
///
/// ```
/// # extern crate argmin;
/// # extern crate ndarray;
/// use argmin::prelude::*;
/// use argmin::solver::newton::NewtonCG;
/// # use argmin::testfunctions::{rosenbrock_2d, rosenbrock_2d_derivative, rosenbrock_2d_hessian};
/// use ndarray::{Array, Array1, Array2};
///
/// # #[derive(Clone)]
/// # struct MyProblem {}
/// #
/// # impl ArgminOperator for MyProblem {
/// #     type Parameters = Array1<f64>;
/// #     type OperatorOutput = f64;
/// #     type Hessian = Array2<f64>;
/// #
/// #     fn apply(&self, p: &Self::Parameters) -> Result<Self::OperatorOutput, Error> {
/// #         Ok(rosenbrock_2d(&p.to_vec(), 1.0, 100.0))
/// #     }
/// #
/// #     fn gradient(&self, p: &Self::Parameters) -> Result<Self::Parameters, Error> {
/// #         Ok(Array1::from_vec(rosenbrock_2d_derivative(&p.to_vec(), 1.0, 100.0)))
/// #     }
/// #
/// #     fn hessian(&self, p: &Self::Parameters) -> Result<Self::Hessian, Error> {
/// #         let h = rosenbrock_2d_hessian(&p.to_vec(), 1.0, 100.0);
/// #         Ok(Array::from_shape_vec((2, 2), h)?)
/// #     }
/// # }
/// #
/// # fn run() -> Result<(), Error> {
/// // Define cost function
/// let cost = MyProblem {};
///
/// // Define initial parameter vector
/// let init_param: Array1<f64> = Array1::from_vec(vec![-1.2, 1.0]);
///
/// // Set up solver
/// let mut solver = NewtonCG::new(cost, init_param);
///
/// // Set maximum number of iterations
/// solver.set_max_iters(20);
///
/// // Attach a logger
/// solver.add_logger(ArgminSlogLogger::term());
///
/// // Run solver
/// solver.run()?;
///
/// // Wait a second (lets the logger flush everything before printing again)
/// std::thread::sleep(std::time::Duration::from_secs(1));
///
/// // Print result
/// println!("{:?}", solver.result());
/// #     Ok(())
/// # }
/// #
/// # fn main() {
/// #     if let Err(ref e) = run() {
/// #         println!("{} {}", e.as_fail(), e.backtrace());
/// #         std::process::exit(1);
/// #     }
/// # }
/// ```
///
/// # References:
///
/// [0] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(ArgminSolver)]
pub struct NewtonCG<'a, T, H, O>
where
    T: 'a
        + Clone
        + Default
        + Send
        + Sync
        + Debug
        + ArgminScaledAdd<T, f64, T>
        + ArgminDot<T, f64>
        + ArgminAdd<T, T>
        + ArgminSub<T, T>
        + ArgminZero
        + ArgminNorm<f64>
        + ArgminMul<f64, T>,
    H: 'a + Clone + Default + Send + Sync + ArgminInv<H> + ArgminDot<T, T>,
    O: 'a + Clone + ArgminOperator<Parameters = T, OperatorOutput = f64, Hessian = H>,
{
    /// line search
    linesearch: Box<ArgminLineSearch<Parameters = T, OperatorOutput = f64, Hessian = H> + 'a>,
    /// curvature_threshold
    curvature_threshold: f64,
    /// Base stuff
    base: ArgminBase<T, f64, H, O>,
}

impl<'a, T, H, O> NewtonCG<'a, T, H, O>
where
    T: 'a
        + Clone
        + Default
        + Send
        + Sync
        + Debug
        + ArgminScaledAdd<T, f64, T>
        + ArgminDot<T, f64>
        + ArgminAdd<T, T>
        + ArgminSub<T, T>
        + ArgminZero
        + ArgminNorm<f64>
        + ArgminMul<f64, T>,
    H: 'a + Clone + Default + Send + Sync + ArgminInv<H> + ArgminDot<T, T>,
    O: 'a + Clone + ArgminOperator<Parameters = T, OperatorOutput = f64, Hessian = H>,
{
    /// Constructor
    pub fn new(cost_function: O, init_param: T) -> Self {
        // let linesearch = HagerZhangLineSearch::new(cost_function.clone());
        let linesearch = MoreThuenteLineSearch::new(cost_function.clone());
        NewtonCG {
            linesearch: Box::new(linesearch),
            curvature_threshold: 0.0,
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

    /// Set curvature threshold
    pub fn set_curvature_threshold(&mut self, threshold: f64) -> &mut Self {
        self.curvature_threshold = threshold;
        self
    }
}

impl<'a, T, H, O> ArgminNextIter for NewtonCG<'a, T, H, O>
where
    T: 'a
        + Clone
        + Default
        + Send
        + Sync
        + Debug
        + ArgminScaledAdd<T, f64, T>
        + ArgminDot<T, f64>
        + ArgminAdd<T, T>
        + ArgminSub<T, T>
        + ArgminZero
        + ArgminNorm<f64>
        + ArgminMul<f64, T>,
    H: 'a + Clone + Send + Sync + Default + ArgminInv<H> + ArgminDot<T, T>,
    O: 'a + Clone + ArgminOperator<Parameters = T, OperatorOutput = f64, Hessian = H>,
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

        let mut x_p = param.zero_like();
        let mut x: T = param.zero_like();
        let mut cg = ConjugateGradient::new(op, grad.mul(&(-1.0)), x_p.clone())?;

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
            let curvature = p.dot(&hessian.dot(&p));
            // println!("iter: {:?}, curv: {:?}", iter, curvature);
            if curvature <= self.curvature_threshold {
                if iter == 0 {
                    x = grad.mul(&(-1.0));
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
    T: 'a + Clone + Send + Sync,
    H: 'a + Clone + Default + ArgminDot<T, T> + Send + Sync,
{
    hessian: H,
    phantom: std::marker::PhantomData<&'a T>,
}

impl<'a, T, H> CGSubProblem<'a, T, H>
where
    T: 'a + Clone + Send + Sync,
    H: 'a + Clone + Default + ArgminDot<T, T> + Send + Sync,
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
    T: 'a + Clone + Send + Sync,
    H: 'a + Clone + Default + ArgminDot<T, T> + Send + Sync,
{
    type Parameters = T;
    type OperatorOutput = T;
    type Hessian = ();

    fn apply(&self, p: &T) -> Result<T, Error> {
        Ok(self.hessian.dot(&p))
    }
}
