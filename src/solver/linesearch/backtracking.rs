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
/// use argmin::solver::linesearch::{BacktrackingLineSearch, ArmijoCondition};
/// # use argmin::testfunctions::{sphere, sphere_derivative};
/// # use serde::{Deserialize, Serialize};
///
/// # #[derive(Clone, Default, Serialize, Deserialize)]
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
/// // Set condition
/// let cond = ArmijoCondition::new(0.5)?;
///
/// // Set up Line Search method
/// let mut solver = BacktrackingLineSearch::new(operator, cond);
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
#[derive(Serialize, Deserialize)]
pub struct BacktrackingLineSearch<P, L> {
    /// initial parameter vector
    init_param: Option<P>,
    /// initial cost
    init_cost: Option<f64>,
    /// initial gradient
    init_grad: Option<P>,
    /// Search direction
    search_direction: Option<P>,
    /// Contraction factor rho
    rho: f64,
    /// Stopping condition
    condition: Box<L>,
    /// alpha
    alpha: f64,
}

impl<P, L> BacktrackingLineSearch<P, L> {
    pub fn new(condition: L) -> Self {
        BacktrackingLineSearch {
            init_param: None,
            init_cost: None,
            init_grad: None,
            search_direction: None,
            rho: 0.9,
            condition: Box::new(condition),
            alpha: 1.0,
        }
    }

    pub fn init_param(mut self, param: P) -> Self {
        self.init_param = Some(param);
        self
    }

    pub fn init_grad(mut self, grad: P) -> Self {
        self.init_grad = Some(grad);
        self
    }

    pub fn search_direction(mut self, direction: P) -> Self {
        self.search_direction = Some(direction);
        self
    }

    pub fn init_cost(mut self, init_cost: f64) -> Self {
        self.init_cost = Some(init_cost);
        self
    }

    pub fn rho(mut self, rho: f64) -> Result<Self, Error> {
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

    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }
}

impl<P, L> ArgminLineSearch<P> for BacktrackingLineSearch<P, L> {
    /// Set search direction
    fn set_search_direction(&mut self, search_direction: P) {
        self.search_direction = Some(search_direction);
    }

    /// Set initial parameter
    fn set_init_param(&mut self, param: P) {
        self.init_param = Some(param);
    }

    /// Set initial alpha value
    fn set_init_alpha(&mut self, alpha: f64) -> Result<(), Error> {
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
    fn set_init_cost(&mut self, init_cost: f64) {
        self.init_cost = Some(init_cost);
    }

    /// Set initial gradient
    fn set_init_grad(&mut self, init_grad: P) {
        self.init_grad = Some(init_grad);
    }
}

impl<O, P, L> Solver<O> for BacktrackingLineSearch<P, L>
where
    P: Clone + Serialize,
    O: ArgminOp<Param = P, Output = f64>,
    P: ArgminSub<P, P> + ArgminDot<P, f64> + ArgminScaledAdd<P, f64, P>,
    L: LineSearchCondition<P>,
{
    fn init(&mut self) -> Result<(), Error> {
        if self.init_param.is_none() {
            return Err(ArgminError::NotInitialized {
                text: "BacktrackingLineSearch: init_param must be set.".to_string(),
            }
            .into());
        }
        if self.init_cost.is_none() {
            return Err(ArgminError::NotInitialized {
                text: "BacktrackingLineSearch: init_cost must be set.".to_string(),
            }
            .into());
        }
        if self.init_grad.is_none() {
            return Err(ArgminError::NotInitialized {
                text: "BacktrackingLineSearch: init_grad must be set.".to_string(),
            }
            .into());
        }
        if self.search_direction.is_none() {
            return Err(ArgminError::NotInitialized {
                text: "BacktrackingLineSearch: search_direction must be set.".to_string(),
            }
            .into());
        }
        Ok(())
    }

    fn next_iter<'a>(
        &mut self,
        op: &mut OpWrapper<'a, O>,
        _state: IterState<P, O::Hessian>,
    ) -> Result<ArgminIterData<P, P>, Error> {
        // this can't go wrong
        let init_param = self.init_param.clone().unwrap();
        let search_direction = self.search_direction.clone().unwrap();

        let new_param = init_param.scaled_add(&self.alpha, &search_direction);

        let cur_cost = op.apply(&new_param)?;

        self.alpha *= self.rho;

        let mut out = ArgminIterData::new(new_param.clone(), cur_cost);

        if self.condition.requires_cur_grad() {
            let grad = op.gradient(&new_param)?;
            out.set_grad(grad);
        }
        Ok(out)
    }

    fn terminate(&mut self, state: &IterState<O::Param, O::Hessian>) -> TerminationReason {
        if self.condition.eval(
            state.cur_cost,
            state.cur_grad.clone(),
            self.init_cost.clone().unwrap(),
            self.init_grad.clone().unwrap(),
            self.search_direction.clone().unwrap(),
            self.alpha,
        ) {
            TerminationReason::LineSearchConditionMet
        } else {
            TerminationReason::NotTerminated
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::send_sync_test;
    use crate::MinimalNoOperator;

    send_sync_test!(backtrackinglinesearch,
                    BacktrackingLineSearch<MinimalNoOperator, ArmijoCondition>);
}
