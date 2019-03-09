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
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

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
#[derive(Serialize, Deserialize)]
pub struct BFGS<L, H> {
    /// Inverse Hessian
    inv_hessian: H,
    /// line search
    linesearch: L,
}

impl<L, H> BFGS<L, H> {
    /// Constructor
    pub fn new(init_inverse_hessian: H, linesearch: L) -> Self {
        BFGS {
            inv_hessian: init_inverse_hessian,
            linesearch: linesearch,
        }
    }
}

impl<O, L, H> Solver<O> for BFGS<L, H>
where
    O: ArgminOp<Output = f64, Hessian = H>,
    O::Param: Debug
        + Default
        + ArgminSub<O::Param, O::Param>
        + ArgminDot<O::Param, f64>
        + ArgminDot<O::Param, O::Hessian>
        + ArgminScaledAdd<O::Param, f64, O::Param>
        + ArgminNorm<f64>
        + ArgminMul<f64, O::Param>,
    O::Hessian: Clone
        + Default
        + Debug
        + Serialize
        + DeserializeOwned
        + ArgminSub<O::Hessian, O::Hessian>
        + ArgminDot<O::Param, O::Param>
        + ArgminDot<O::Hessian, O::Hessian>
        + ArgminAdd<O::Hessian, O::Hessian>
        + ArgminMul<f64, O::Hessian>
        + ArgminTranspose
        + ArgminEye,
    L: Clone + ArgminLineSearch<O::Param> + Solver<OpWrapper<O>>,
{
    fn init(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &IterState<O>,
    ) -> Result<Option<ArgminIterData<O>>, Error> {
        let param = state.get_param();
        let cost = op.apply(&param)?;
        let grad = op.gradient(&param)?;
        Ok(Some(
            ArgminIterData::new().param(param).cost(cost).grad(grad),
        ))
    }

    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &IterState<O>,
    ) -> Result<ArgminIterData<O>, Error> {
        let param = state.get_param();
        let cur_cost = state.get_cost();
        let prev_grad = if let Some(grad) = state.get_grad() {
            grad
        } else {
            op.gradient(&param)?
        };

        let p = self.inv_hessian.dot(&prev_grad).mul(&(-1.0));

        self.linesearch.set_search_direction(p);

        // Run solver
        let linesearch_result = Executor::new(
            OpWrapper::new_from_op(&op),
            self.linesearch.clone(),
            param.clone(),
        )
        .grad(prev_grad.clone())
        .cost(cur_cost)
        .run_fast()?;

        // take care of function eval counts
        op.consume_op(linesearch_result.operator);

        let xk1 = linesearch_result.param;

        let grad = op.gradient(&xk1)?;
        let yk = grad.sub(&prev_grad);

        let sk = xk1.sub(&param);

        let yksk: f64 = yk.dot(&sk);
        let rhok = 1.0 / yksk;

        let e = self.inv_hessian.eye_like();
        let mat1: O::Hessian = sk.dot(&yk);
        let mat1 = mat1.mul(&rhok);

        let mat2 = mat1.clone().t();

        let tmp1 = e.sub(&mat1);
        let tmp2 = e.sub(&mat2);

        let sksk: O::Hessian = sk.dot(&sk);
        let sksk = sksk.mul(&rhok);

        // if self.cur_iter() == 0 {
        //     let ykyk: f64 = yk.dot(&yk);
        //     self.inv_hessian = self.inv_hessian.eye_like().mul(&(yksk / ykyk));
        //     println!("{:?}", self.inv_hessian);
        // }

        self.inv_hessian = tmp1.dot(&self.inv_hessian.dot(&tmp2)).add(&sksk);

        Ok(ArgminIterData::new()
            .param(xk1)
            .cost(linesearch_result.cost)
            .grad(grad))
    }

    fn terminate(&mut self, state: &IterState<O>) -> TerminationReason {
        if state.get_grad().unwrap().norm() < std::f64::EPSILON.sqrt() {
            return TerminationReason::TargetPrecisionReached;
        }
        if (state.get_prev_cost() - state.get_cost()).abs() < std::f64::EPSILON {
            return TerminationReason::NoChangeInCost;
        }
        TerminationReason::NotTerminated
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
