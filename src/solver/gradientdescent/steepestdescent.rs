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

use prelude::*;
use solver::linesearch::HagerZhangLineSearch;
use std;
use std::default::Default;

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
/// # use argmin::testfunctions::{rosenbrock_2d, rosenbrock_2d_derivative};
///
/// # #[derive(Clone)]
/// # struct MyProblem {}
/// #
/// # impl ArgminOperator for MyProblem {
/// #     type Parameters = Vec<f64>;
/// #     type OperatorOutput = f64;
/// #     type Hessian = ();
/// #
/// #     fn apply(&self, p: &Self::Parameters) -> Result<Self::OperatorOutput, Error> {
/// #         Ok(rosenbrock_2d(p, 1.0, 100.0))
/// #     }
/// #
/// #     fn gradient(&self, p: &Self::Parameters) -> Result<Self::Parameters, Error> {
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
/// // Pick a line search. If no line search algorithm is provided, SteepestDescent defaults to
/// // HagerZhang.
/// let linesearch = HagerZhangLineSearch::new(&cost);
/// // let linesearch = MoreThuenteLineSearch::new(&cost);
/// // let linesearch = BacktrackingLineSearch::new(&cost);
///
/// // Set up solver
/// let mut solver = SteepestDescent::new(&cost, init_param)?;
/// // Set linesearch. This can be omitted, which will then default to `HagerZhangLineSearch`
/// solver.set_linesearch(Box::new(linesearch));
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
    base: ArgminBase<'a, T, f64, H>,
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
        cost_function: &'a ArgminOperator<Parameters = T, OperatorOutput = f64, Hessian = H>,
        init_param: T,
    ) -> Result<Self, Error> {
        let linesearch = HagerZhangLineSearch::new(cost_function);
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
        self.linesearch.base_reset();

        let param_new = self.cur_param();
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
