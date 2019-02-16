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
use serde::{Deserialize, Serialize};

/// BFGS method
///
/// # Example
///
/// ```rust
/// # extern crate argmin;
/// # extern crate ndarray;
/// use argmin::prelude::*;
/// use argmin::solver::quasinewton::BFGS;
/// use argmin::solver::linesearch::MoreThuenteLineSearch;
/// # use argmin::testfunctions::{rosenbrock_2d, rosenbrock_2d_derivative};
/// use ndarray::{array, Array1, Array2};
/// # use serde::{Deserialize, Serialize};
///
/// # #[derive(Clone, Default, Serialize, Deserialize)]
/// # struct MyProblem { }
/// #
/// #  impl ArgminOp for MyProblem {
/// #      type Param = Array1<f64>;
/// #      type Output = f64;
/// #      type Hessian = Array2<f64>;
/// #
/// #      fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
/// #          Ok(rosenbrock_2d(&p.to_vec(), 1.0, 100.0))
/// #      }
/// #
/// #      fn gradient(&self, p: &Self::Param) -> Result<Self::Param, Error> {
/// #          Ok(Array1::from_vec(rosenbrock_2d_derivative(
/// #              &p.to_vec(),
/// #              1.0,
/// #              100.0,
/// #          )))
/// #      }
/// #  }
/// #
/// #  fn run() -> Result<(), Error> {
/// // Define cost function
/// let cost = MyProblem {};
///
/// // Define initial parameter vector
/// // let init_param: Array1<f64> = Array1::from_vec(vec![1.2, 1.2]);
/// let init_param: Array1<f64> = array![-1.2, 1.0];
/// let init_hessian: Array2<f64> = Array2::eye(2);
///
/// // set up a line search
/// let linesearch = MoreThuenteLineSearch::new(cost.clone());
///
/// // Set up solver
/// let mut solver = BFGS::new(cost, init_param, init_hessian, linesearch);
///
/// // Set maximum number of iterations
/// solver.set_max_iters(80);
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
/// # Ok(())
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
#[stop("self.cur_grad().norm() < std::f64::EPSILON.sqrt()" => TargetPrecisionReached)]
pub struct BFGS<O, L>
where
    O: ArgminOp<Output = f64>,
    O::Param: ArgminSub<O::Param, O::Param>
        + ArgminDot<O::Param, f64>
        + ArgminDot<O::Param, O::Hessian>
        + ArgminScaledAdd<O::Param, f64, O::Param>
        + ArgminNorm<f64>
        + ArgminMul<f64, O::Param>,
    O::Hessian: ArgminSub<O::Hessian, O::Hessian>
        + ArgminDot<O::Param, O::Param>
        + ArgminDot<O::Hessian, O::Hessian>
        + ArgminAdd<O::Hessian, O::Hessian>
        + ArgminMul<f64, O::Hessian>
        + ArgminTranspose
        + ArgminEye,
    L: ArgminLineSearch<Param = O::Param, Output = O::Output, Hessian = O::Hessian>,
{
    /// Inverse Hessian
    inv_hessian: O::Hessian,
    /// line search
    linesearch: Box<L>,
    /// Base stuff
    base: ArgminBase<O>,
}

impl<O, L> BFGS<O, L>
where
    O: ArgminOp<Output = f64>,
    O::Param: ArgminSub<O::Param, O::Param>
        + ArgminDot<O::Param, f64>
        + ArgminDot<O::Param, O::Hessian>
        + ArgminScaledAdd<O::Param, f64, O::Param>
        + ArgminNorm<f64>
        + ArgminMul<f64, O::Param>,
    O::Hessian: ArgminSub<O::Hessian, O::Hessian>
        + ArgminDot<O::Param, O::Param>
        + ArgminDot<O::Hessian, O::Hessian>
        + ArgminAdd<O::Hessian, O::Hessian>
        + ArgminMul<f64, O::Hessian>
        + ArgminTranspose
        + ArgminEye,
    L: ArgminLineSearch<Param = O::Param, Output = O::Output, Hessian = O::Hessian>,
{
    /// Constructor
    pub fn new(
        cost_function: O,
        init_param: O::Param,
        init_inverse_hessian: O::Hessian,
        linesearch: L,
    ) -> Self {
        BFGS {
            inv_hessian: init_inverse_hessian,
            linesearch: Box::new(linesearch),
            base: ArgminBase::new(cost_function, init_param),
        }
    }
}

impl<O, L> ArgminIter for BFGS<O, L>
where
    O: ArgminOp<Output = f64>,
    O::Param: ArgminSub<O::Param, O::Param>
        + ArgminDot<O::Param, f64>
        + ArgminDot<O::Param, O::Hessian>
        + ArgminScaledAdd<O::Param, f64, O::Param>
        + ArgminNorm<f64>
        + ArgminMul<f64, O::Param>,
    O::Hessian: ArgminSub<O::Hessian, O::Hessian>
        + ArgminDot<O::Param, O::Param>
        + ArgminDot<O::Hessian, O::Hessian>
        + ArgminAdd<O::Hessian, O::Hessian>
        + ArgminMul<f64, O::Hessian>
        + ArgminTranspose
        + ArgminEye,
    L: ArgminLineSearch<Param = O::Param, Output = O::Output, Hessian = O::Hessian>,
{
    type Param = O::Param;
    type Output = O::Output;
    type Hessian = O::Hessian;

    fn init(&mut self) -> Result<(), Error> {
        let cost = self.apply(&self.base.cur_param())?;
        let grad = self.gradient(&self.base.cur_param())?;
        self.base.set_cur_grad(grad);
        self.base.set_cur_cost(cost);
        Ok(())
    }

    fn next_iter(&mut self) -> Result<ArgminIterData<Self::Param>, Error> {
        // reset line search
        self.linesearch.base_reset();

        let param = self.cur_param();
        let cur_cost = self.cur_cost();
        let prev_grad = self.base.cur_grad();
        let p = self.inv_hessian.dot(&prev_grad).mul(&(-1.0));

        self.linesearch.set_initial_parameter(param.clone());
        self.linesearch.set_initial_gradient(prev_grad.clone());
        self.linesearch.set_initial_cost(cur_cost);
        self.linesearch
            .set_search_direction(p.mul(&(1.0 / p.norm())));
        // self.linesearch.set_search_direction(p);
        self.linesearch.run_fast()?;

        let linesearch_result = self.linesearch.result();
        let xk1 = linesearch_result.param;

        let grad = self.gradient(&xk1)?;
        let yk = grad.sub(&prev_grad);
        self.base.set_cur_grad(grad);

        let sk = xk1.sub(&param);

        let yksk: f64 = yk.dot(&sk);
        let rhok = 1.0 / yksk;

        let e = self.inv_hessian.eye_like();
        let mat1: Self::Hessian = sk.dot(&yk);
        let mat1 = mat1.mul(&rhok);

        let mat2 = mat1.clone().t();

        let tmp1 = e.sub(&mat1);
        let tmp2 = e.sub(&mat2);

        let sksk: Self::Hessian = sk.dot(&sk);
        let sksk = sksk.mul(&rhok);

        self.inv_hessian = tmp1.dot(&self.inv_hessian.dot(&tmp2)).add(&sksk);

        let out = ArgminIterData::new(xk1, linesearch_result.cost);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::send_sync_test;
    use crate::solver::linesearch::MoreThuenteLineSearch;

    type Operator = MinimalNoOperator;

    send_sync_test!(bfgs, BFGS<Operator, MoreThuenteLineSearch<Operator>>);
}
