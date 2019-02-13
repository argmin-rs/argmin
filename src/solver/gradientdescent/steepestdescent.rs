// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Steepest Descent method
//!
//! [SteepestDescent](struct.SteepestDescent.html)
//!
//! # References:
//!
//! [0] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
//! Springer. ISBN 0-387-30303-0.

use crate::prelude::*;
use serde::{Deserialize, Serialize};

/// Steepest descent iteratively takes steps in the direction of the strongest negative gradient.
/// In each iteration, a line search is employed to obtain an appropriate step length.
///
/// # Example
///
/// ```rust
/// # #![allow(unused_imports)]
/// #
/// # extern crate argmin;
/// use argmin::prelude::*;
/// use argmin::solver::gradientdescent::SteepestDescent;
/// use argmin::solver::linesearch::HagerZhangLineSearch;
/// use argmin::solver::linesearch::MoreThuenteLineSearch;
/// use argmin::solver::linesearch::BacktrackingLineSearch;
/// # use serde::{Deserialize, Serialize};
/// # use argmin::testfunctions::{rosenbrock_2d, rosenbrock_2d_derivative};
///
/// # #[derive(Clone, Default, Serialize, Deserialize)]
/// # struct MyProblem {}
/// #
/// # impl ArgminOp for MyProblem {
/// #     type Param = Vec<f64>;
/// #     type Output = f64;
/// #     type Hessian = ();
/// #
/// #     fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
/// #         Ok(rosenbrock_2d(p, 1.0, 100.0))
/// #     }
/// #
/// #     fn gradient(&self, p: &Self::Param) -> Result<Self::Param, Error> {
/// #         Ok(rosenbrock_2d_derivative(p, 1.0, 100.0))
/// #     }
/// # }
/// #
/// # fn run() -> Result<(), Error> {
/// // Define cost function (must implement `ArgminOperator`)
/// let cost = MyProblem { };
///
/// // Define initial parameter vector
/// let init_param: Vec<f64> = vec![1.2, 1.2];
///
/// // Pick a line search.
/// // let linesearch = HagerZhangLineSearch::new(cost.clone());
/// let linesearch = MoreThuenteLineSearch::new(cost.clone());
/// // let linesearch = BacktrackingLineSearch::new(cost.clone());
///
/// // Set up solver
/// let mut solver = SteepestDescent::new(cost, init_param, linesearch)?;
///
/// // Set maximum number of iterations
/// solver.set_max_iters(100);
///
/// // Attach a logger which will output information in each iteration.
/// solver.add_logger(ArgminSlogLogger::term_noblock());
///
/// // Run the solver
/// solver.run()?;
///
/// // Wait a second (lets the logger flush everything first)
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
pub struct SteepestDescent<O, L>
where
    O: ArgminOp<Output = f64>,
    O::Param: ArgminSub<O::Param, O::Param>
        + ArgminDot<O::Param, f64>
        + ArgminScaledAdd<O::Param, f64, O::Param>
        + ArgminMul<f64, O::Param>
        + ArgminSub<O::Param, O::Param>
        + ArgminNorm<f64>,
    L: ArgminLineSearch<Param = O::Param, Output = f64, Hessian = O::Hessian>,
{
    /// line search
    linesearch: Box<L>,
    /// Base stuff
    base: ArgminBase<O>,
}

impl<O, L> SteepestDescent<O, L>
where
    O: ArgminOp<Output = f64>,
    O::Param: ArgminSub<O::Param, O::Param>
        + ArgminDot<O::Param, f64>
        + ArgminScaledAdd<O::Param, f64, O::Param>
        + ArgminMul<f64, O::Param>
        + ArgminSub<O::Param, O::Param>
        + ArgminNorm<f64>,
    L: ArgminLineSearch<Param = O::Param, Output = f64, Hessian = O::Hessian>,
{
    /// Constructor
    pub fn new(cost_function: O, init_param: O::Param, linesearch: L) -> Result<Self, Error> {
        Ok(SteepestDescent {
            linesearch: Box::new(linesearch),
            base: ArgminBase::new(cost_function, init_param),
        })
    }
}

impl<O, L> ArgminIter for SteepestDescent<O, L>
where
    O: ArgminOp<Output = f64>,
    O::Param: ArgminSub<O::Param, O::Param>
        + ArgminDot<O::Param, f64>
        + ArgminScaledAdd<O::Param, f64, O::Param>
        + ArgminMul<f64, O::Param>
        + ArgminSub<O::Param, O::Param>
        + ArgminNorm<f64>,
    L: ArgminLineSearch<Param = O::Param, Output = f64, Hessian = O::Hessian>,
{
    type Param = O::Param;
    type Output = f64;
    type Hessian = O::Hessian;

    /// Perform one iteration of SA algorithm
    fn next_iter(&mut self) -> Result<ArgminIterData<Self::Param>, Error> {
        // reset line search
        self.linesearch.base_reset();

        let param_new = self.cur_param();
        let new_cost = self.apply(&param_new)?;
        let new_grad = self.gradient(&param_new)?;

        let norm = new_grad.norm();

        self.linesearch.set_initial_parameter(param_new);
        self.linesearch.set_initial_gradient(new_grad.clone());
        self.linesearch.set_initial_cost(new_cost);
        self.linesearch
            .set_search_direction(new_grad.mul(&(-1.0 / norm)));

        self.linesearch.run_fast()?;

        let linesearch_result = self.linesearch.result();

        let out = ArgminIterData::new(linesearch_result.param, linesearch_result.cost);
        Ok(out)
    }
}
