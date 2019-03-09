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
/// use argmin::solver::linesearch::MoreThuenteLineSearch;
/// use ndarray::{Array, Array1, Array2};
///
/// # use serde::{Deserialize, Serialize};
/// #
/// # #[derive(Clone, Default, Serialize, Deserialize)]
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
/// // set up line search
/// let linesearch = MoreThuenteLineSearch::new(cost.clone());
///
/// // Set up solver
/// let mut solver = NewtonCG::new(cost, init_param, linesearch);
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
#[derive(Serialize, Deserialize)]
pub struct NewtonCG<L> {
    /// line search
    linesearch: L,
    /// curvature_threshold
    curvature_threshold: f64,
}

impl<L> NewtonCG<L> {
    /// Constructor
    pub fn new(linesearch: L) -> Self {
        NewtonCG {
            linesearch: linesearch,
            curvature_threshold: 0.0,
        }
    }

    /// Set curvature threshold
    pub fn curvature_threshold(mut self, threshold: f64) -> Self {
        self.curvature_threshold = threshold;
        self
    }
}

impl<O, L> Solver<O> for NewtonCG<L>
where
    O: ArgminOp<Output = f64>,
    O::Param: Send
        + Sync
        + Clone
        + Serialize
        + Default
        + ArgminSub<O::Param, O::Param>
        + ArgminAdd<O::Param, O::Param>
        + ArgminDot<O::Param, f64>
        + ArgminScaledAdd<O::Param, f64, O::Param>
        + ArgminMul<f64, O::Param>
        + ArgminZero
        + ArgminNorm<f64>,
    O::Hessian: Send
        + Sync
        + Clone
        + Serialize
        + Default
        + ArgminInv<O::Hessian>
        + ArgminDot<O::Param, O::Param>,
    L: Clone + ArgminLineSearch<O::Param> + Solver<OpWrapper<O>>,
{
    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &IterState<O>,
    ) -> Result<ArgminIterData<O>, Error> {
        let param = state.get_param();
        let grad = op.gradient(&param)?;
        let hessian = op.hessian(&param)?;

        // Solve CG subproblem
        let cg_op: CGSubProblem<O::Param, O::Hessian> = CGSubProblem::new(hessian.clone());
        let mut cg_op = OpWrapper::new(&cg_op);

        let mut x_p = param.zero_like();
        let mut x: O::Param = param.zero_like();
        let mut cg = ConjugateGradient::new(grad.mul(&(-1.0)))?;

        let mut cg_state = IterState::new(x_p.clone());
        cg.init(&mut cg_op, &cg_state)?;
        let grad_norm = grad.norm();
        for iter in 0.. {
            let data = cg.next_iter(&mut cg_op, &cg_state)?;
            x = data.get_param().unwrap();
            let p = cg.p_prev();
            let curvature = p.dot(&hessian.dot(&p));
            if curvature <= self.curvature_threshold {
                if iter == 0 {
                    x = grad.mul(&(-1.0));
                    break;
                } else {
                    x = x_p;
                    break;
                }
            }
            if data.get_cost().unwrap() <= (0.5f64).min(grad_norm.sqrt()) * grad_norm {
                break;
            }
            cg_state.param(x.clone());
            cg_state.cost(data.get_cost().unwrap());
            x_p = x.clone();
        }

        // take care of counting
        op.consume_op(cg_op);

        // perform line search
        self.linesearch.set_search_direction(x);

        // Run solver
        let linesearch_result =
            Executor::new(OpWrapper::new_from_op(&op), self.linesearch.clone(), param)
                .grad(grad)
                .cost(state.get_cost())
                .run_fast()?;

        op.consume_op(linesearch_result.operator);

        Ok(ArgminIterData::new()
            .param(linesearch_result.param)
            .cost(linesearch_result.cost))
    }

    fn terminate(&mut self, state: &IterState<O>) -> TerminationReason {
        if (state.get_cost() - state.get_prev_cost()).abs() < std::f64::EPSILON {
            TerminationReason::NoChangeInCost
        } else {
            TerminationReason::NotTerminated
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::send_sync_test;
    use crate::solver::linesearch::MoreThuenteLineSearch;

    // Only works with ndarray feature because of the required inverse of a matrix
    #[cfg(feature = "ndarrayl")]
    type Operator = NoOperator<ndarray::Array1<f64>, f64, ndarray::Array2<f64>>;

    // Only works with ndarray feature because of the required inverse of a matrix
    #[cfg(feature = "ndarrayl")]
    send_sync_test!(newton_cg, NewtonCG<Operator, MoreThuenteLineSearch<Operator>>);

    send_sync_test!(cg_subproblem, CGSubProblem<Vec<f64>, Vec<Vec<f64>>>);
}
