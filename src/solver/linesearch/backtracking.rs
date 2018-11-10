// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Line search methods
//!
//! TODO: Proper documentation.
//!
//! # References:
//!
//! [0] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
//! Springer. ISBN 0-387-30303-0.
// //!
// //! # Example
// //!
// //! ```rust
// //! todo
// //! ```

use prelude::*;
use solver::linesearch::condition::*;
use std;

/// Backtracking Line Search
#[derive(ArgminSolver)]
#[stop("self.eval_condition()" => LineSearchConditionMet)]
pub struct BacktrackingLineSearch<'a, T, H>
where
    T: std::default::Default
        + Clone
        + ArgminSub<T>
        + ArgminDot<T, f64>
        + ArgminScaledAdd<T, f64>
        + ArgminScaledSub<T, f64>,
    H: Clone + std::default::Default,
{
    /// initial parameter vector
    init_param: T,
    /// initial cost
    init_cost: f64,
    /// initial gradient
    init_grad: T,
    /// Search direction
    search_direction: T,
    /// rho
    rho: f64,
    /// Stopping condition
    condition: Box<LineSearchCondition<T>>,
    /// alpha
    alpha: f64,
    /// base
    base: ArgminBase<'a, T, f64, H>,
}

impl<'a, T, H> BacktrackingLineSearch<'a, T, H>
where
    T: std::default::Default
        + Clone
        + ArgminSub<T>
        + ArgminDot<T, f64>
        + ArgminScaledAdd<T, f64>
        + ArgminScaledSub<T, f64>,
    H: Clone + std::default::Default,
    BacktrackingLineSearch<'a, T, H>: ArgminSolver<Parameters = T, OperatorOutput = f64>,
{
    /// Constructor
    ///
    /// Parameters:
    ///
    /// `cost_function`: cost function
    /// `rho`: todo
    pub fn new(
        operator: &'a ArgminOperator<Parameters = T, OperatorOutput = f64, Hessian = H>,
    ) -> Self {
        // let cond = ArmijoCondition::new(0.0001).unwrap();
        // let cond = WolfeCondition::new(0.0001, 0.9).unwrap();
        let cond = StrongWolfeCondition::new(0.0001, 0.9).unwrap();
        // let cond = GoldsteinCondition::new(0.25).unwrap();
        BacktrackingLineSearch {
            init_param: T::default(),
            init_cost: std::f64::INFINITY,
            init_grad: T::default(),
            search_direction: T::default(),
            rho: 0.9,
            condition: Box::new(cond),
            alpha: 1.0,
            base: ArgminBase::new(operator, T::default()),
        }
    }

    /// set current gradient value
    pub fn set_cur_grad(&mut self, grad: T) -> &mut Self {
        self.base.set_cur_grad(grad);
        self
    }

    /// Set rho
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

    /// Set condition
    pub fn set_condition(&mut self, condition: Box<LineSearchCondition<T>>) -> &mut Self {
        self.condition = condition;
        self
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

impl<'a, T, H> ArgminLineSearch for BacktrackingLineSearch<'a, T, H>
where
    T: std::default::Default
        + Clone
        + ArgminSub<T>
        + ArgminDot<T, f64>
        + ArgminScaledAdd<T, f64>
        + ArgminScaledSub<T, f64>,
    H: Clone + std::default::Default,
    BacktrackingLineSearch<'a, T, H>: ArgminSolver<Parameters = T, OperatorOutput = f64>,
{
    /// Set search direction
    fn set_search_direction(&mut self, search_direction: T) {
        self.search_direction = search_direction;
    }

    /// Set initial parameter
    fn set_initial_parameter(&mut self, param: T) {
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
    fn set_initial_gradient(&mut self, init_grad: T) {
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

impl<'a, T, H> ArgminNextIter for BacktrackingLineSearch<'a, T, H>
where
    T: std::default::Default
        + Clone
        + ArgminSub<T>
        + ArgminDot<T, f64>
        + ArgminScaledAdd<T, f64>
        + ArgminScaledSub<T, f64>,
    H: Clone + std::default::Default,
{
    type Parameters = T;
    type OperatorOutput = f64;
    type Hessian = H;

    fn next_iter(&mut self) -> Result<ArgminIterationData<Self::Parameters>, Error> {
        let new_param = self
            .init_param
            .scaled_add(self.alpha, self.search_direction.clone());

        let cur_cost = self.apply(&new_param)?;

        if self.condition.requires_cur_grad() {
            let grad = self.gradient(&new_param)?;
            self.set_cur_grad(grad);
        }

        self.alpha *= self.rho;

        let out = ArgminIterationData::new(new_param, cur_cost);
        Ok(out)
    }
}
