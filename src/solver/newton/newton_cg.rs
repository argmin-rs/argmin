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
use serde::{Deserialize, Serialize};

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
/// # #[derive(Clone, Default)]
/// # struct MyProblem {}
/// #
/// # impl ArgminOp for MyProblem {
/// #     type Param = Array1<f64>;
/// #     type Output = f64;
/// #     type Hessian = Array2<f64>;
/// #
/// #     fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
/// #         Ok(rosenbrock_2d(&p.to_vec(), 1.0, 100.0))
/// #     }
/// #
/// #     fn gradient(&self, p: &Self::Param) -> Result<Self::Param, Error> {
/// #         Ok(Array1::from_vec(rosenbrock_2d_derivative(&p.to_vec(), 1.0, 100.0)))
/// #     }
/// #
/// #     fn hessian(&self, p: &Self::Param) -> Result<Self::Hessian, Error> {
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
#[derive(ArgminSolver, Serialize, Deserialize)]
pub struct NewtonCG<O, L>
where
    O: ArgminOp<Output = f64>,
    <O as ArgminOp>::Param: ArgminSub<<O as ArgminOp>::Param, <O as ArgminOp>::Param>
        + ArgminAdd<<O as ArgminOp>::Param, <O as ArgminOp>::Param>
        + ArgminDot<<O as ArgminOp>::Param, f64>
        + ArgminScaledAdd<<O as ArgminOp>::Param, f64, <O as ArgminOp>::Param>
        + ArgminMul<f64, <O as ArgminOp>::Param>
        + ArgminZero
        + ArgminNorm<f64>,
    <O as ArgminOp>::Hessian: ArgminInv<<O as ArgminOp>::Hessian>
        + ArgminDot<<O as ArgminOp>::Param, <O as ArgminOp>::Param>,
    L: ArgminLineSearch<Param = O::Param, Output = O::Output, Hessian = O::Hessian>,
{
    /// line search
    linesearch: Box<L>,
    /// curvature_threshold
    curvature_threshold: f64,
    /// Base stuff
    base: ArgminBase<O>,
}

impl<O, L> NewtonCG<O, L>
where
    O: ArgminOp<Output = f64>,
    <O as ArgminOp>::Param: ArgminSub<<O as ArgminOp>::Param, <O as ArgminOp>::Param>
        + ArgminAdd<<O as ArgminOp>::Param, <O as ArgminOp>::Param>
        + ArgminDot<<O as ArgminOp>::Param, f64>
        + ArgminScaledAdd<<O as ArgminOp>::Param, f64, <O as ArgminOp>::Param>
        + ArgminMul<f64, <O as ArgminOp>::Param>
        + ArgminZero
        + ArgminNorm<f64>,
    <O as ArgminOp>::Hessian: ArgminInv<<O as ArgminOp>::Hessian>
        + ArgminDot<<O as ArgminOp>::Param, <O as ArgminOp>::Param>,
    L: ArgminLineSearch<Param = O::Param, Output = O::Output, Hessian = O::Hessian>,
{
    /// Constructor
    pub fn new(cost_function: O, init_param: <O as ArgminOp>::Param, linesearch: L) -> Self {
        NewtonCG {
            linesearch: Box::new(linesearch),
            curvature_threshold: 0.0,
            base: ArgminBase::new(cost_function, init_param),
        }
    }

    /// Set curvature threshold
    pub fn set_curvature_threshold(&mut self, threshold: f64) -> &mut Self {
        self.curvature_threshold = threshold;
        self
    }
}

impl<O, L> ArgminIter for NewtonCG<O, L>
where
    O: ArgminOp<Output = f64>,
    <O as ArgminOp>::Param: ArgminSub<<O as ArgminOp>::Param, <O as ArgminOp>::Param>
        + ArgminAdd<<O as ArgminOp>::Param, <O as ArgminOp>::Param>
        + ArgminDot<<O as ArgminOp>::Param, f64>
        + ArgminScaledAdd<<O as ArgminOp>::Param, f64, <O as ArgminOp>::Param>
        + ArgminMul<f64, <O as ArgminOp>::Param>
        + ArgminZero
        + ArgminNorm<f64>,
    <O as ArgminOp>::Hessian: ArgminInv<<O as ArgminOp>::Hessian>
        + ArgminDot<<O as ArgminOp>::Param, <O as ArgminOp>::Param>,
    L: ArgminLineSearch<Param = O::Param, Output = O::Output, Hessian = O::Hessian>,
{
    type Param = <O as ArgminOp>::Param;
    type Output = <O as ArgminOp>::Output;
    type Hessian = <O as ArgminOp>::Hessian;

    fn next_iter(&mut self) -> Result<ArgminIterData<Self::Param>, Error> {
        let param = self.cur_param();
        let grad = self.gradient(&param)?;
        let hessian = self.hessian(&param)?;

        // Solve CG subproblem
        let op: CGSubProblem<Self::Param, Self::Hessian> = CGSubProblem::new(hessian.clone());

        let mut x_p = param.zero_like();
        let mut x: Self::Param = param.zero_like();
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

        let out = ArgminIterData::new(linesearch_result.param, linesearch_result.cost);
        Ok(out)
    }
}

#[derive(Clone, Default, Serialize, Deserialize)]
struct CGSubProblem<T, H> {
    hessian: H,
    phantom: std::marker::PhantomData<T>,
}

impl<T, H> CGSubProblem<T, H>
where
    T: Clone + Send + Sync,
    H: Clone + Default + ArgminDot<T, T> + Send + Sync,
{
    /// constructor
    pub fn new(hessian: H) -> Self {
        CGSubProblem {
            hessian,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<T, H> ArgminOp for CGSubProblem<T, H>
where
    T: Clone + Default + Send + Sync + Serialize + serde::de::DeserializeOwned,
    H: Clone + Default + ArgminDot<T, T> + Send + Sync + Serialize + serde::de::DeserializeOwned,
{
    type Param = T;
    type Output = T;
    type Hessian = ();

    fn apply(&self, p: &T) -> Result<T, Error> {
        Ok(self.hessian.dot(&p))
    }
}
