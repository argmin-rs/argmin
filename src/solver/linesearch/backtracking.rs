// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! * [Backtracking line search](struct.BacktrackingLineSearch.html)

use crate::prelude::*;
use crate::solver::linesearch::condition::*;
use serde::{Deserialize, Serialize};

/// The Backtracking line search is a simple method to find a step length which obeys the Armijo
/// (sufficient decrease) condition.
///
/// # Example
///
/// ```
/// # extern crate argmin;
/// use argmin::prelude::*;
/// use argmin::solver::linesearch::BacktrackingLineSearch;
/// # use argmin::testfunctions::{sphere, sphere_derivative};
///
/// # #[derive(Clone, Default)]
/// # struct Sphere {}
/// #
/// # impl ArgminOp for Sphere {
/// #     type Param = Vec<f64>;
/// #     type Output = f64;
/// #     type Hessian = ();
/// #
/// #     fn apply(&self, param: &Vec<f64>) -> Result<f64, Error> {
/// #         Ok(sphere(param))
/// #     }
/// #
/// #     fn gradient(&self, param: &Vec<f64>) -> Result<Vec<f64>, Error> {
/// #         Ok(sphere_derivative(param))
/// #     }
/// # }
/// #
/// # fn run() -> Result<(), Error> {
/// // definie inital parameter vector
/// let init_param: Vec<f64> = vec![1.0, 0.0];
///
/// // Define problem
/// let operator = Sphere {};
///
/// // Set up Line Search method
/// let mut solver = BacktrackingLineSearch::new(operator);
///
/// // Set search direction
/// solver.set_search_direction(vec![-2.0, 0.0]);
///
/// // Set initial position
/// solver.set_initial_parameter(init_param);
///
/// // Set contraction factor
/// solver.set_rho(0.9)?;
///
/// // Calculate initial cost ...
/// solver.calc_initial_cost()?;
///
/// // ... or, alternatively, set cost if it is already computed
/// // solver.set_initial_cost(...);
///
/// // Calculate initial gradient ...
/// solver.calc_initial_gradient()?;
///
/// // .. or, alternatively, set gradient if it is already computed
/// // solver.set_initial_gradient(...);
///
/// // Set initial step length
/// solver.set_initial_alpha(1.0)?;
///
/// // Set maximum number of iterations
/// solver.set_max_iters(100);
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
/// #     }
/// # }
/// ```
///
/// # References:
///
/// [0] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
///
/// [1] Wikipedia: https://en.wikipedia.org/wiki/Backtracking_line_search
#[derive(ArgminSolver, Serialize, Deserialize)]
#[stop("self.eval_condition()" => LineSearchConditionMet)]
pub struct BacktrackingLineSearch<O, L>
where
    O: ArgminOp<Output = f64>,
    <O as ArgminOp>::Param: ArgminSub<<O as ArgminOp>::Param, <O as ArgminOp>::Param>
        + ArgminDot<<O as ArgminOp>::Param, f64>
        + ArgminScaledAdd<<O as ArgminOp>::Param, f64, <O as ArgminOp>::Param>,
    L: LineSearchCondition<O::Param>,
{
    /// initial parameter vector
    init_param: <O as ArgminOp>::Param,
    /// initial cost
    init_cost: f64,
    /// initial gradient
    init_grad: <O as ArgminOp>::Param,
    /// Search direction
    search_direction: <O as ArgminOp>::Param,
    /// Contraction factor rho
    rho: f64,
    /// Stopping condition
    condition: Box<L>,
    /// alpha
    alpha: f64,
    /// base
    base: ArgminBase<O>,
}

impl<O, L> BacktrackingLineSearch<O, L>
where
    O: ArgminOp<Output = f64>,
    <O as ArgminOp>::Param: ArgminSub<<O as ArgminOp>::Param, <O as ArgminOp>::Param>
        + ArgminDot<<O as ArgminOp>::Param, f64>
        + ArgminScaledAdd<<O as ArgminOp>::Param, f64, <O as ArgminOp>::Param>,
    L: LineSearchCondition<O::Param>,
{
    /// Constructor
    ///
    /// Parameters:
    ///
    /// `operator`: Must implement `ArgminOp`
    pub fn new(operator: O, condition: L) -> Self {
        BacktrackingLineSearch {
            init_param: <O as ArgminOp>::Param::default(),
            init_cost: std::f64::INFINITY,
            init_grad: <O as ArgminOp>::Param::default(),
            search_direction: <O as ArgminOp>::Param::default(),
            rho: 0.9,
            condition: Box::new(condition),
            alpha: 1.0,
            base: ArgminBase::new(operator, <O as ArgminOp>::Param::default()),
        }
    }

    /// set current gradient value
    pub fn set_cur_grad(&mut self, grad: <O as ArgminOp>::Param) -> &mut Self {
        self.base.set_cur_grad(grad);
        self
    }

    /// Set contraction factor rho
    pub fn set_rho(&mut self, rho: f64) -> Result<&mut Self, Error> {
        if rho <= 0.0 || rho >= 1.0 {
            return Err(ArgminError::InvalidParameter {
                text: "BacktrackingLineSearch: Contraction factor rho must be in (0, 1)."
                    .to_string(),
            }
            .into());
        }
        self.rho = rho;
        Ok(self)
    }

    fn eval_condition(&self) -> bool {
        self.condition.eval(
            self.cur_cost(),
            self.cur_grad(),
            self.init_cost,
            self.init_grad.clone(),
            self.search_direction.clone(),
            self.alpha,
        )
    }
}

impl<O, L> ArgminLineSearch for BacktrackingLineSearch<O, L>
where
    O: ArgminOp<Output = f64>,
    <O as ArgminOp>::Param: ArgminSub<<O as ArgminOp>::Param, <O as ArgminOp>::Param>
        + ArgminDot<<O as ArgminOp>::Param, f64>
        + ArgminScaledAdd<<O as ArgminOp>::Param, f64, <O as ArgminOp>::Param>,
    L: LineSearchCondition<O::Param>,
{
    /// Set search direction
    fn set_search_direction(&mut self, search_direction: <O as ArgminOp>::Param) {
        self.search_direction = search_direction;
    }

    /// Set initial parameter
    fn set_initial_parameter(&mut self, param: <O as ArgminOp>::Param) {
        self.init_param = param.clone();
        self.set_cur_param(param);
    }

    /// Set initial alpha value
    fn set_initial_alpha(&mut self, alpha: f64) -> Result<(), Error> {
        if alpha <= 0.0 {
            return Err(ArgminError::InvalidParameter {
                text: "LineSearch: Inital alpha must be > 0.".to_string(),
            }
            .into());
        }
        self.alpha = alpha;
        Ok(())
    }

    /// Set initial cost function value
    fn set_initial_cost(&mut self, init_cost: f64) {
        self.init_cost = init_cost;
    }

    /// Set initial gradient
    fn set_initial_gradient(&mut self, init_grad: <O as ArgminOp>::Param) {
        self.init_grad = init_grad;
    }

    /// Calculate initial cost function value
    fn calc_initial_cost(&mut self) -> Result<(), Error> {
        let tmp = self.cur_param();
        self.init_cost = self.apply(&tmp)?;
        Ok(())
    }

    /// Calculate initial cost function value
    fn calc_initial_gradient(&mut self) -> Result<(), Error> {
        let tmp = self.cur_param();
        self.init_grad = self.gradient(&tmp)?;
        Ok(())
    }
}

impl<O, L> ArgminIter for BacktrackingLineSearch<O, L>
where
    O: ArgminOp<Output = f64>,
    <O as ArgminOp>::Param: ArgminSub<<O as ArgminOp>::Param, <O as ArgminOp>::Param>
        + ArgminDot<<O as ArgminOp>::Param, f64>
        + ArgminScaledAdd<<O as ArgminOp>::Param, f64, <O as ArgminOp>::Param>,
    L: LineSearchCondition<O::Param>,
{
    type Param = <O as ArgminOp>::Param;
    type Output = f64;
    type Hessian = <O as ArgminOp>::Hessian;

    fn next_iter(&mut self) -> Result<ArgminIterData<Self::Param>, Error> {
        let new_param = self
            .init_param
            .scaled_add(&self.alpha, &self.search_direction);

        let cur_cost = self.apply(&new_param)?;

        if self.condition.requires_cur_grad() {
            let grad = self.gradient(&new_param)?;
            self.set_cur_grad(grad);
        }

        self.alpha *= self.rho;

        let out = ArgminIterData::new(new_param, cur_cost);
        Ok(out)
    }
}
