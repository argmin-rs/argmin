// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::{ArgminFloat, Problem, State, TerminationReason};
use instant;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Maintains the state from iteration to iteration of a solver
///
/// This struct is passed from one iteration of an algorithm to the next.
///
/// Keeps track of
///
/// * parameter vector of current and previous iteration
/// * best parameter vector of current and previous iteration
/// * gradient of current and previous iteration
/// * Jacobian of current and previous iteration
/// * Hessian of current and previous iteration
/// * inverse Hessian of current and previous iteration
/// * cost function value of current and previous iteration
/// * current and previous best cost function value
/// * target cost function value
/// * current iteration number
/// * iteration number where the last best parameter vector was found
/// * maximum number of iterations that will be executed
/// * problem function evaluation counts (cost function, gradient, jacobian, hessian,
///   annealing,...)
/// * elapsed time
/// * termination reason (set to [`TerminationReason::NotTerminated`] if not terminated yet)
#[derive(Clone, Default, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct IterState<P, G, J, H, F> {
    /// Current parameter vector
    pub param: Option<P>,
    /// Previous parameter vector
    pub prev_param: Option<P>,
    /// Current best parameter vector
    pub best_param: Option<P>,
    /// Previous best parameter vector
    pub prev_best_param: Option<P>,
    /// Current cost function value
    pub cost: F,
    /// Previous cost function value
    pub prev_cost: F,
    /// Current best cost function value
    pub best_cost: F,
    /// Previous best cost function value
    pub prev_best_cost: F,
    /// Target cost function value
    pub target_cost: F,
    /// Current gradient
    pub grad: Option<G>,
    /// Previous gradient
    pub prev_grad: Option<G>,
    /// Current Hessian
    pub hessian: Option<H>,
    /// Previous Hessian
    pub prev_hessian: Option<H>,
    /// Current inverse Hessian
    pub inv_hessian: Option<H>,
    /// Previous inverse Hessian
    pub prev_inv_hessian: Option<H>,
    /// Current Jacobian
    pub jacobian: Option<J>,
    /// Previous Jacobian
    pub prev_jacobian: Option<J>,
    /// Current iteration
    pub iter: u64,
    /// Iteration number of last best cost
    pub last_best_iter: u64,
    /// Maximum number of iterations
    pub max_iters: u64,
    /// Evaluation counts
    pub counts: HashMap<String, u64>,
    /// Time required so far
    pub time: Option<instant::Duration>,
    /// Reason of termination
    pub termination_reason: TerminationReason,
}

impl<P, G, J, H, F> IterState<P, G, J, H, F>
where
    Self: State<Float = F>,
    F: ArgminFloat,
{
    /// Set parameter vector. This shifts the stored parameter vector to the previous parameter
    /// vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State};
    /// # let state: IterState<Vec<f64>, (), (), (), f64> = IterState::new();
    /// # let param_old = vec![1.0f64, 2.0f64];
    /// # let state = state.param(param_old);
    /// # assert!(state.prev_param.is_none());
    /// # assert_eq!(state.param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # let param = vec![0.0f64, 3.0f64];
    /// let state = state.param(param);
    /// # assert_eq!(state.prev_param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(state.param.as_ref().unwrap()[0].to_ne_bytes(), 0.0f64.to_ne_bytes());
    /// # assert_eq!(state.param.as_ref().unwrap()[1].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// ```
    #[must_use]
    pub fn param(mut self, param: P) -> Self {
        std::mem::swap(&mut self.prev_param, &mut self.param);
        self.param = Some(param);
        self
    }

    /// Set gradient. This shifts the stored gradient to the previous gradient.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State};
    /// # let state: IterState<(), Vec<f64>, (), (), f64> = IterState::new();
    /// # let grad_old = vec![1.0f64, 2.0f64];
    /// # let state = state.gradient(grad_old);
    /// # assert!(state.prev_grad.is_none());
    /// # assert_eq!(state.grad.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.grad.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # let grad = vec![0.0f64, 3.0f64];
    /// let state = state.gradient(grad);
    /// # assert_eq!(state.prev_grad.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_grad.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(state.grad.as_ref().unwrap()[0].to_ne_bytes(), 0.0f64.to_ne_bytes());
    /// # assert_eq!(state.grad.as_ref().unwrap()[1].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// ```
    #[must_use]
    pub fn gradient(mut self, gradient: G) -> Self {
        std::mem::swap(&mut self.prev_grad, &mut self.grad);
        self.grad = Some(gradient);
        self
    }

    /// Set Hessian. This shifts the stored Hessian to the previous Hessian.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State};
    /// # let state: IterState<(), (), (), Vec<f64>, f64> = IterState::new();
    /// # let hessian_old = vec![1.0f64, 2.0f64];
    /// # let state = state.hessian(hessian_old);
    /// # assert!(state.prev_hessian.is_none());
    /// # assert_eq!(state.hessian.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.hessian.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # let hessian = vec![0.0f64, 3.0f64];
    /// let state = state.hessian(hessian);
    /// # assert_eq!(state.prev_hessian.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_hessian.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(state.hessian.as_ref().unwrap()[0].to_ne_bytes(), 0.0f64.to_ne_bytes());
    /// # assert_eq!(state.hessian.as_ref().unwrap()[1].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// ```
    #[must_use]
    pub fn hessian(mut self, hessian: H) -> Self {
        std::mem::swap(&mut self.prev_hessian, &mut self.hessian);
        self.hessian = Some(hessian);
        self
    }

    /// Set inverse Hessian. This shifts the stored inverse Hessian to the previous inverse Hessian.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State};
    /// # let state: IterState<(), (), (), Vec<f64>, f64> = IterState::new();
    /// # let inv_hessian_old = vec![1.0f64, 2.0f64];
    /// # let state = state.inv_hessian(inv_hessian_old);
    /// # assert!(state.prev_inv_hessian.is_none());
    /// # assert_eq!(state.inv_hessian.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.inv_hessian.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # let inv_hessian = vec![0.0f64, 3.0f64];
    /// let state = state.inv_hessian(inv_hessian);
    /// # assert_eq!(state.prev_inv_hessian.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_inv_hessian.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(state.inv_hessian.as_ref().unwrap()[0].to_ne_bytes(), 0.0f64.to_ne_bytes());
    /// # assert_eq!(state.inv_hessian.as_ref().unwrap()[1].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// ```
    #[must_use]
    pub fn inv_hessian(mut self, inv_hessian: H) -> Self {
        std::mem::swap(&mut self.prev_inv_hessian, &mut self.inv_hessian);
        self.inv_hessian = Some(inv_hessian);
        self
    }

    /// Set Jacobian. This shifts the stored Jacobian to the previous Jacobian.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State};
    /// # let state: IterState<(), (), Vec<f64>, (), f64> = IterState::new();
    /// # let jacobian_old = vec![1.0f64, 2.0f64];
    /// # let state = state.jacobian(jacobian_old);
    /// # assert!(state.prev_jacobian.is_none());
    /// # assert_eq!(state.jacobian.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.jacobian.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # let jacobian = vec![0.0f64, 3.0f64];
    /// let state = state.jacobian(jacobian);
    /// # assert_eq!(state.prev_jacobian.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_jacobian.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(state.jacobian.as_ref().unwrap()[0].to_ne_bytes(), 0.0f64.to_ne_bytes());
    /// # assert_eq!(state.jacobian.as_ref().unwrap()[1].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// ```
    #[must_use]
    pub fn jacobian(mut self, jacobian: J) -> Self {
        std::mem::swap(&mut self.prev_jacobian, &mut self.jacobian);
        self.jacobian = Some(jacobian);
        self
    }

    /// Set the current cost function value. This shifts the stored cost function value to the
    /// previous cost function value.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State};
    /// # let state: IterState<(), (), Vec<f64>, (), f64> = IterState::new();
    /// # let cost_old = 1.0f64;
    /// # let state = state.cost(cost_old);
    /// # assert_eq!(state.prev_cost.to_ne_bytes(), f64::INFINITY.to_ne_bytes());
    /// # assert_eq!(state.cost.to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # let cost = 0.0f64;
    /// let state = state.cost(cost);
    /// # assert_eq!(state.prev_cost.to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.cost.to_ne_bytes(), 0.0f64.to_ne_bytes());
    /// ```
    #[must_use]
    pub fn cost(mut self, cost: F) -> Self {
        std::mem::swap(&mut self.prev_cost, &mut self.cost);
        self.cost = cost;
        self
    }

    /// Set target cost.
    ///
    /// When this cost is reached, the algorithm will stop. The default is
    /// `Self::Float::NEG_INFINITY`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State, ArgminFloat};
    /// # let state: IterState<Vec<f64>, (), (), (), f64> = IterState::new();
    /// # assert_eq!(state.target_cost.to_ne_bytes(), f64::NEG_INFINITY.to_ne_bytes());
    /// let state = state.target_cost(0.0);
    /// # assert_eq!(state.target_cost.to_ne_bytes(), 0.0f64.to_ne_bytes());
    /// ```
    #[must_use]
    pub fn target_cost(mut self, target_cost: F) -> Self {
        self.target_cost = target_cost;
        self
    }

    /// Set maximum number of iterations
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State, ArgminFloat};
    /// # let state: IterState<Vec<f64>, (), (), (), f64> = IterState::new();
    /// # assert_eq!(state.max_iters, std::u64::MAX);
    /// let state = state.max_iters(1000);
    /// # assert_eq!(state.max_iters, 1000);
    /// ```
    #[must_use]
    pub fn max_iters(mut self, iters: u64) -> Self {
        self.max_iters = iters;
        self
    }

    /// Returns the current cost function value
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State, ArgminFloat};
    /// # let state: IterState<Vec<f64>, (), (), (), f64> = IterState::new();
    /// # let state = state.cost(2.0);
    /// let cost = state.get_cost();
    /// # assert_eq!(cost.to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn get_cost(&self) -> F {
        self.cost
    }

    /// Returns the previous cost function value
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State, ArgminFloat};
    /// # let mut state: IterState<Vec<f64>, (), (), (), f64> = IterState::new();
    /// # state.prev_cost = 2.0;
    /// let prev_cost = state.get_prev_cost();
    /// # assert_eq!(prev_cost.to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn get_prev_cost(&self) -> F {
        self.prev_cost
    }

    /// Returns the current best cost function value
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State, ArgminFloat};
    /// # let mut state: IterState<Vec<f64>, (), (), (), f64> = IterState::new();
    /// # state.best_cost = 2.0;
    /// let best_cost = state.get_best_cost();
    /// # assert_eq!(best_cost.to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn get_best_cost(&self) -> F {
        self.best_cost
    }

    /// Returns the previous best cost function value
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State, ArgminFloat};
    /// # let mut state: IterState<Vec<f64>, (), (), (), f64> = IterState::new();
    /// # state.prev_best_cost = 2.0;
    /// let prev_best_cost = state.get_prev_best_cost();
    /// # assert_eq!(prev_best_cost.to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn get_prev_best_cost(&self) -> F {
        self.prev_best_cost
    }

    /// Returns the target cost function value
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State, ArgminFloat};
    /// # let mut state: IterState<Vec<f64>, (), (), (), f64> = IterState::new();
    /// # assert_eq!(state.target_cost.to_ne_bytes(), std::f64::NEG_INFINITY.to_ne_bytes());
    /// # state.target_cost = 0.0;
    /// let target_cost = state.get_target_cost();
    /// # assert_eq!(target_cost.to_ne_bytes(), 0.0f64.to_ne_bytes());
    /// ```
    pub fn get_target_cost(&self) -> F {
        self.target_cost
    }

    /// Moves the current parameter vector out and replaces it internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State, ArgminFloat};
    /// # let mut state: IterState<Vec<f64>, (), (), (), f64> = IterState::new();
    /// # assert!(state.take_param().is_none());
    /// # let mut state = state.param(vec![1.0, 2.0]);
    /// # assert_eq!(state.param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let param = state.take_param();  // Option<P>
    /// # assert!(state.take_param().is_none());
    /// # assert!(state.param.is_none());
    /// # assert_eq!(param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn take_param(&mut self) -> Option<P> {
        self.param.take()
    }

    /// Returns a reference to previous parameter vector
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State, ArgminFloat};
    /// # let mut state: IterState<Vec<f64>, (), (), (), f64> = IterState::new();
    /// # assert!(state.prev_param.is_none());
    /// # state.prev_param = Some(vec![1.0, 2.0]);
    /// # assert_eq!(state.prev_param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let prev_param = state.get_prev_param();  // Option<&P>
    /// # assert_eq!(prev_param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(prev_param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn get_prev_param(&self) -> Option<&P> {
        self.prev_param.as_ref()
    }

    /// Moves the previous parameter vector out and replaces it internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State, ArgminFloat};
    /// # let mut state: IterState<Vec<f64>, (), (), (), f64> = IterState::new();
    /// # assert!(state.take_prev_param().is_none());
    /// # state.prev_param = Some(vec![1.0, 2.0]);
    /// # assert_eq!(state.prev_param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let prev_param = state.take_prev_param();  // Option<P>
    /// # assert!(state.take_prev_param().is_none());
    /// # assert!(state.prev_param.is_none());
    /// # assert_eq!(prev_param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(prev_param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn take_prev_param(&mut self) -> Option<P> {
        self.prev_param.take()
    }

    /// Returns a reference to previous best parameter vector
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State, ArgminFloat};
    /// # let mut state: IterState<Vec<f64>, (), (), (), f64> = IterState::new();
    /// # assert!(state.prev_best_param.is_none());
    /// # state.prev_best_param = Some(vec![1.0, 2.0]);
    /// # assert_eq!(state.prev_best_param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_best_param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let prev_best_param = state.get_prev_best_param();  // Option<&P>
    /// # assert_eq!(prev_best_param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(prev_best_param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn get_prev_best_param(&self) -> Option<&P> {
        self.prev_best_param.as_ref()
    }

    /// Moves the best parameter vector out and replaces it internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State, ArgminFloat};
    /// # let mut state: IterState<Vec<f64>, (), (), (), f64> = IterState::new();
    /// # assert!(state.take_best_param().is_none());
    /// # state.best_param = Some(vec![1.0, 2.0]);
    /// # assert_eq!(state.best_param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.best_param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let best_param = state.take_best_param();  // Option<P>
    /// # assert!(state.take_best_param().is_none());
    /// # assert!(state.best_param.is_none());
    /// # assert_eq!(best_param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(best_param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn take_best_param(&mut self) -> Option<P> {
        self.best_param.take()
    }

    /// Moves the previous best parameter vector out and replaces it internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State, ArgminFloat};
    /// # let mut state: IterState<Vec<f64>, (), (), (), f64> = IterState::new();
    /// # assert!(state.take_prev_best_param().is_none());
    /// # state.prev_best_param = Some(vec![1.0, 2.0]);
    /// # assert_eq!(state.prev_best_param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_best_param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let prev_best_param = state.take_prev_best_param();  // Option<P>
    /// # assert!(state.take_prev_best_param().is_none());
    /// # assert!(state.prev_best_param.is_none());
    /// # assert_eq!(prev_best_param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(prev_best_param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn take_prev_best_param(&mut self) -> Option<P> {
        self.prev_best_param.take()
    }

    /// Returns a reference to the gradient
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State, ArgminFloat};
    /// # let mut state: IterState<(), Vec<f64>, (), (), f64> = IterState::new();
    /// # assert!(state.grad.is_none());
    /// # assert!(state.get_gradient().is_none());
    /// # state.grad = Some(vec![1.0, 2.0]);
    /// # assert_eq!(state.grad.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.grad.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let grad = state.get_gradient();  // Option<&G>
    /// # assert_eq!(grad.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(grad.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn get_gradient(&self) -> Option<&G> {
        self.grad.as_ref()
    }

    /// Moves the gradient out and replaces it internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State, ArgminFloat};
    /// # let mut state: IterState<(), Vec<f64>, (), (), f64> = IterState::new();
    /// # assert!(state.take_gradient().is_none());
    /// # state.grad = Some(vec![1.0, 2.0]);
    /// # assert_eq!(state.grad.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.grad.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let grad = state.take_gradient();  // Option<G>
    /// # assert!(state.take_gradient().is_none());
    /// # assert!(state.grad.is_none());
    /// # assert_eq!(grad.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(grad.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn take_gradient(&mut self) -> Option<G> {
        self.grad.take()
    }

    /// Returns a reference to the previous gradient
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State, ArgminFloat};
    /// # let mut state: IterState<(), Vec<f64>, (), (), f64> = IterState::new();
    /// # assert!(state.prev_grad.is_none());
    /// # assert!(state.get_prev_gradient().is_none());
    /// # state.prev_grad = Some(vec![1.0, 2.0]);
    /// # assert_eq!(state.prev_grad.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_grad.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let prev_grad = state.get_prev_gradient();  // Option<&G>
    /// # assert_eq!(prev_grad.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(prev_grad.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn get_prev_gradient(&self) -> Option<&G> {
        self.prev_grad.as_ref()
    }

    /// Moves the gradient out and replaces it internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State, ArgminFloat};
    /// # let mut state: IterState<(), Vec<f64>, (), (), f64> = IterState::new();
    /// # assert!(state.take_prev_gradient().is_none());
    /// # state.prev_grad = Some(vec![1.0, 2.0]);
    /// # assert_eq!(state.prev_grad.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_grad.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let prev_grad = state.take_prev_gradient();  // Option<G>
    /// # assert!(state.take_prev_gradient().is_none());
    /// # assert!(state.prev_grad.is_none());
    /// # assert_eq!(prev_grad.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(prev_grad.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    pub fn take_prev_gradient(&mut self) -> Option<G> {
        self.prev_grad.take()
    }

    /// Returns a reference to the current Hessian
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State, ArgminFloat};
    /// # let mut state: IterState<(), (), (), Vec<Vec<f64>>, f64> = IterState::new();
    /// # assert!(state.hessian.is_none());
    /// # assert!(state.get_hessian().is_none());
    /// # state.hessian = Some(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    /// # assert_eq!(state.hessian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.hessian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(state.hessian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(state.hessian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// let hessian = state.get_hessian();  // Option<&H>
    /// # assert_eq!(hessian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(hessian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(hessian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(hessian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// ```
    pub fn get_hessian(&self) -> Option<&H> {
        self.hessian.as_ref()
    }

    /// Moves the Hessian out and replaces it internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State, ArgminFloat};
    /// # let mut state: IterState<(), (), (), Vec<Vec<f64>>, f64> = IterState::new();
    /// # assert!(state.hessian.is_none());
    /// # assert!(state.take_hessian().is_none());
    /// # state.hessian = Some(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    /// # assert_eq!(state.hessian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.hessian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(state.hessian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(state.hessian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// let hessian = state.take_hessian();  // Option<H>
    /// # assert!(state.take_hessian().is_none());
    /// # assert!(state.hessian.is_none());
    /// # assert_eq!(hessian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(hessian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(hessian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(hessian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// ```
    pub fn take_hessian(&mut self) -> Option<H> {
        self.hessian.take()
    }

    /// Returns a reference to the previous Hessian
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State, ArgminFloat};
    /// # let mut state: IterState<(), (), (), Vec<Vec<f64>>, f64> = IterState::new();
    /// # assert!(state.prev_hessian.is_none());
    /// # assert!(state.get_prev_hessian().is_none());
    /// # state.prev_hessian = Some(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    /// # assert_eq!(state.prev_hessian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_hessian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_hessian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_hessian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// let prev_hessian = state.get_prev_hessian();  // Option<&H>
    /// # assert_eq!(prev_hessian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(prev_hessian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(prev_hessian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(prev_hessian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// ```
    pub fn get_prev_hessian(&self) -> Option<&H> {
        self.prev_hessian.as_ref()
    }

    /// Moves the previous Hessian out and replaces it internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State, ArgminFloat};
    /// # let mut state: IterState<(), (), (), Vec<Vec<f64>>, f64> = IterState::new();
    /// # assert!(state.prev_hessian.is_none());
    /// # assert!(state.take_prev_hessian().is_none());
    /// # state.prev_hessian = Some(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    /// # assert_eq!(state.prev_hessian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_hessian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_hessian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_hessian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// let prev_hessian = state.take_prev_hessian();  // Option<H>
    /// # assert!(state.take_prev_hessian().is_none());
    /// # assert!(state.prev_hessian.is_none());
    /// # assert_eq!(prev_hessian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(prev_hessian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(prev_hessian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(prev_hessian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// ```
    pub fn take_prev_hessian(&mut self) -> Option<H> {
        self.prev_hessian.take()
    }

    /// Returns a reference to the current inverse Hessian
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State, ArgminFloat};
    /// # let mut state: IterState<(), (), (), Vec<Vec<f64>>, f64> = IterState::new();
    /// # assert!(state.inv_hessian.is_none());
    /// # assert!(state.get_inv_hessian().is_none());
    /// # state.inv_hessian = Some(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    /// # assert_eq!(state.inv_hessian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.inv_hessian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(state.inv_hessian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(state.inv_hessian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// let inv_hessian = state.get_inv_hessian();  // Option<&H>
    /// # assert_eq!(inv_hessian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(inv_hessian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(inv_hessian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(inv_hessian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// ```
    pub fn get_inv_hessian(&self) -> Option<&H> {
        self.inv_hessian.as_ref()
    }

    /// Moves the inverse Hessian out and replaces it internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State, ArgminFloat};
    /// # let mut state: IterState<(), (), (), Vec<Vec<f64>>, f64> = IterState::new();
    /// # assert!(state.inv_hessian.is_none());
    /// # assert!(state.take_inv_hessian().is_none());
    /// # state.inv_hessian = Some(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    /// # assert_eq!(state.inv_hessian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.inv_hessian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(state.inv_hessian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(state.inv_hessian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// let inv_hessian = state.take_inv_hessian();  // Option<H>
    /// # assert!(state.take_inv_hessian().is_none());
    /// # assert!(state.inv_hessian.is_none());
    /// # assert_eq!(inv_hessian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(inv_hessian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(inv_hessian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(inv_hessian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// ```
    pub fn take_inv_hessian(&mut self) -> Option<H> {
        self.inv_hessian.take()
    }

    /// Returns a reference to the previous inverse Hessian
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State, ArgminFloat};
    /// # let mut state: IterState<(), (), (), Vec<Vec<f64>>, f64> = IterState::new();
    /// # assert!(state.prev_inv_hessian.is_none());
    /// # assert!(state.get_prev_inv_hessian().is_none());
    /// # state.prev_inv_hessian = Some(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    /// # assert_eq!(state.prev_inv_hessian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_inv_hessian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_inv_hessian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_inv_hessian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// let prev_inv_hessian = state.get_prev_inv_hessian();  // Option<&H>
    /// # assert_eq!(prev_inv_hessian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(prev_inv_hessian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(prev_inv_hessian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(prev_inv_hessian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// ```
    pub fn get_prev_inv_hessian(&self) -> Option<&H> {
        self.prev_inv_hessian.as_ref()
    }

    /// Moves the previous Hessian out and replaces it internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State, ArgminFloat};
    /// # let mut state: IterState<(), (), (), Vec<Vec<f64>>, f64> = IterState::new();
    /// # assert!(state.prev_inv_hessian.is_none());
    /// # assert!(state.take_prev_inv_hessian().is_none());
    /// # state.prev_inv_hessian = Some(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    /// # assert_eq!(state.prev_inv_hessian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_inv_hessian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_inv_hessian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_inv_hessian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// let prev_inv_hessian = state.take_prev_inv_hessian();  // Option<H>
    /// # assert!(state.take_prev_inv_hessian().is_none());
    /// # assert!(state.prev_inv_hessian.is_none());
    /// # assert_eq!(prev_inv_hessian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(prev_inv_hessian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(prev_inv_hessian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(prev_inv_hessian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// ```
    pub fn take_prev_inv_hessian(&mut self) -> Option<H> {
        self.prev_inv_hessian.take()
    }

    /// Returns a reference to the current Jacobian
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State, ArgminFloat};
    /// # let mut state: IterState<(), (), Vec<Vec<f64>>, (), f64> = IterState::new();
    /// # assert!(state.jacobian.is_none());
    /// # assert!(state.get_jacobian().is_none());
    /// # state.jacobian = Some(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    /// # assert_eq!(state.jacobian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.jacobian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(state.jacobian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(state.jacobian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// let jacobian = state.get_jacobian();  // Option<&J>
    /// # assert_eq!(jacobian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(jacobian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(jacobian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(jacobian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// ```
    pub fn get_jacobian(&self) -> Option<&J> {
        self.jacobian.as_ref()
    }

    /// Moves the Jacobian out and replaces it internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State, ArgminFloat};
    /// # let mut state: IterState<(), (), Vec<Vec<f64>>, (), f64> = IterState::new();
    /// # assert!(state.jacobian.is_none());
    /// # assert!(state.take_jacobian().is_none());
    /// # state.jacobian = Some(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    /// # assert_eq!(state.jacobian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.jacobian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(state.jacobian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(state.jacobian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// let jacobian = state.take_jacobian();  // Option<J>
    /// # assert!(state.take_jacobian().is_none());
    /// # assert!(state.jacobian.is_none());
    /// # assert_eq!(jacobian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(jacobian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(jacobian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(jacobian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// ```
    pub fn take_jacobian(&mut self) -> Option<J> {
        self.jacobian.take()
    }

    /// Returns a reference to the previous Jacobian
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State, ArgminFloat};
    /// # let mut state: IterState<(), (), Vec<Vec<f64>>, (), f64> = IterState::new();
    /// # assert!(state.prev_jacobian.is_none());
    /// # assert!(state.get_prev_jacobian().is_none());
    /// # state.prev_jacobian = Some(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    /// # assert_eq!(state.prev_jacobian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_jacobian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_jacobian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_jacobian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// let prev_jacobian = state.get_prev_jacobian();  // Option<&J>
    /// # assert_eq!(prev_jacobian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(prev_jacobian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(prev_jacobian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(prev_jacobian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// ```
    pub fn get_prev_jacobian(&self) -> Option<&J> {
        self.prev_jacobian.as_ref()
    }

    /// Moves the previous Jacobian out and replaces it internally with `None`
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State, ArgminFloat};
    /// # let mut state: IterState<(), (), Vec<Vec<f64>>, (), f64> = IterState::new();
    /// # assert!(state.prev_jacobian.is_none());
    /// # assert!(state.take_prev_jacobian().is_none());
    /// # state.prev_jacobian = Some(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    /// # assert_eq!(state.prev_jacobian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_jacobian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_jacobian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(state.prev_jacobian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// let prev_jacobian = state.take_prev_jacobian();  // Option<J>
    /// # assert!(state.take_prev_jacobian().is_none());
    /// # assert!(state.prev_jacobian.is_none());
    /// # assert_eq!(prev_jacobian.as_ref().unwrap()[0][0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(prev_jacobian.as_ref().unwrap()[0][1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// # assert_eq!(prev_jacobian.as_ref().unwrap()[1][0].to_ne_bytes(), 3.0f64.to_ne_bytes());
    /// # assert_eq!(prev_jacobian.as_ref().unwrap()[1][1].to_ne_bytes(), 4.0f64.to_ne_bytes());
    /// ```
    pub fn take_prev_jacobian(&mut self) -> Option<J> {
        self.prev_jacobian.take()
    }
}

impl<P, G, J, H, F> State for IterState<P, G, J, H, F>
where
    P: Clone,
    F: ArgminFloat,
{
    /// Type of parameter vector
    type Param = P;
    /// Floating point precision
    type Float = F;

    /// Create a new IterState instance
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate instant;
    /// # use instant;
    /// # use argmin::core::{IterState, State, ArgminFloat, TerminationReason};
    /// let state: IterState<Vec<f64>, Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>, f64> = IterState::new();
    /// # assert!(state.param.is_none());
    /// # assert!(state.prev_param.is_none());
    /// # assert!(state.best_param.is_none());
    /// # assert!(state.prev_best_param.is_none());
    /// # assert_eq!(state.cost.to_ne_bytes(), f64::INFINITY.to_ne_bytes());
    /// # assert_eq!(state.prev_cost.to_ne_bytes(), f64::INFINITY.to_ne_bytes());
    /// # assert_eq!(state.best_cost.to_ne_bytes(), f64::INFINITY.to_ne_bytes());
    /// # assert_eq!(state.prev_best_cost.to_ne_bytes(), f64::INFINITY.to_ne_bytes());
    /// # assert_eq!(state.target_cost.to_ne_bytes(), f64::NEG_INFINITY.to_ne_bytes());
    /// # assert!(state.grad.is_none());
    /// # assert!(state.prev_grad.is_none());
    /// # assert!(state.hessian.is_none());
    /// # assert!(state.prev_hessian.is_none());
    /// # assert!(state.inv_hessian.is_none());
    /// # assert!(state.prev_inv_hessian.is_none());
    /// # assert!(state.jacobian.is_none());
    /// # assert!(state.prev_jacobian.is_none());
    /// # assert_eq!(state.iter, 0);
    /// # assert_eq!(state.last_best_iter, 0);
    /// # assert_eq!(state.max_iters, std::u64::MAX);
    /// # assert_eq!(state.counts.len(), 0);
    /// # assert_eq!(state.time.unwrap(), instant::Duration::new(0, 0));
    /// # assert_eq!(state.termination_reason, TerminationReason::NotTerminated);
    /// ```
    fn new() -> Self {
        IterState {
            param: None,
            prev_param: None,
            best_param: None,
            prev_best_param: None,
            cost: F::infinity(),
            prev_cost: F::infinity(),
            best_cost: F::infinity(),
            prev_best_cost: F::infinity(),
            target_cost: F::neg_infinity(),
            grad: None,
            prev_grad: None,
            hessian: None,
            prev_hessian: None,
            inv_hessian: None,
            prev_inv_hessian: None,
            jacobian: None,
            prev_jacobian: None,
            iter: 0,
            last_best_iter: 0,
            max_iters: std::u64::MAX,
            counts: HashMap::new(),
            time: Some(instant::Duration::new(0, 0)),
            termination_reason: TerminationReason::NotTerminated,
        }
    }

    /// Checks if the current parameter vector is better than the previous best parameter value. If
    /// a new best parameter vector was found, the state is updated accordingly.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State, ArgminFloat};
    /// let mut state: IterState<Vec<f64>, (), (), (), f64> = IterState::new();
    ///
    /// // Simulating a new, better parameter vector
    /// state.best_param = Some(vec![1.0f64]);
    /// state.best_cost = 10.0;
    /// state.param = Some(vec![2.0f64]);
    /// state.cost = 5.0;
    ///
    /// // Calling update
    /// state.update();
    ///
    /// // Check if update was successful
    /// assert_eq!(state.best_param.as_ref().unwrap()[0], 2.0f64);
    /// assert_eq!(state.best_cost.to_ne_bytes(), state.best_cost.to_ne_bytes());
    /// assert!(state.is_best());
    /// ```
    ///
    /// For algorithms which do not compute the cost function, every new parameter vector will be
    /// the new best:
    ///
    /// ```
    /// # use argmin::core::{IterState, State, ArgminFloat};
    /// let mut state: IterState<Vec<f64>, (), (), (), f64> = IterState::new();
    ///
    /// // Simulating a new, better parameter vector
    /// state.best_param = Some(vec![1.0f64]);
    /// state.param = Some(vec![2.0f64]);
    ///
    /// // Calling update
    /// state.update();
    ///
    /// // Check if update was successful
    /// assert_eq!(state.best_param.as_ref().unwrap()[0], 2.0f64);
    /// assert_eq!(state.best_cost.to_ne_bytes(), state.best_cost.to_ne_bytes());
    /// assert!(state.is_best());
    /// ```
    fn update(&mut self) {
        // check if parameters are the best so far
        // Comparison is done using `<` to avoid new solutions with the same cost function value as
        // the current best to be accepted. However, some solvers to not compute the cost function
        // value (such as the Newton method). Those will always have `Inf` cost. Therefore if both
        // the new value and the previous best value are `Inf`, the solution is also accepted. Care
        // is taken that both `Inf` also have the same sign.
        if self.cost < self.best_cost
            || (self.cost.is_infinite()
                && self.best_cost.is_infinite()
                && self.cost.is_sign_positive() == self.best_cost.is_sign_positive())
        {
            // If there is no parameter vector, then also don't set the best param.
            if let Some(param) = self.param.as_ref().cloned() {
                std::mem::swap(&mut self.prev_best_param, &mut self.best_param);
                self.best_param = Some(param);
            }
            std::mem::swap(&mut self.prev_best_cost, &mut self.best_cost);
            self.best_cost = self.cost;
            self.last_best_iter = self.iter;
        }
    }

    /// Returns a reference to the current parameter vector
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State, ArgminFloat};
    /// # let mut state: IterState<Vec<f64>, (), (), (), f64> = IterState::new();
    /// # assert!(state.param.is_none());
    /// # state.param = Some(vec![1.0, 2.0]);
    /// # assert_eq!(state.param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let param = state.get_param();  // Option<&P>
    /// # assert_eq!(param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    fn get_param(&self) -> Option<&P> {
        self.param.as_ref()
    }

    /// Returns a reference to the current best parameter vector
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State, ArgminFloat};
    /// # let mut state: IterState<Vec<f64>, (), (), (), f64> = IterState::new();
    /// # assert!(state.best_param.is_none());
    /// # state.best_param = Some(vec![1.0, 2.0]);
    /// # assert_eq!(state.best_param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(state.best_param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// let best_param = state.get_best_param();  // Option<&P>
    /// # assert_eq!(best_param.as_ref().unwrap()[0].to_ne_bytes(), 1.0f64.to_ne_bytes());
    /// # assert_eq!(best_param.as_ref().unwrap()[1].to_ne_bytes(), 2.0f64.to_ne_bytes());
    /// ```
    fn get_best_param(&self) -> Option<&P> {
        self.best_param.as_ref()
    }

    /// Sets the termination reason (default: [`TerminationReason::NotTerminated`])
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State, ArgminFloat, TerminationReason};
    /// # let mut state: IterState<Vec<f64>, (), (), (), f64> = IterState::new();
    /// # assert_eq!(state.termination_reason, TerminationReason::NotTerminated);
    /// let state = state.terminate_with(TerminationReason::MaxItersReached);
    /// # assert_eq!(state.termination_reason, TerminationReason::MaxItersReached);
    /// ```
    fn terminate_with(mut self, reason: TerminationReason) -> Self {
        self.termination_reason = reason;
        self
    }

    /// Sets the time required so far.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate instant;
    /// # use instant;
    /// # use argmin::core::{IterState, State, ArgminFloat, TerminationReason};
    /// # let mut state: IterState<Vec<f64>, (), (), (), f64> = IterState::new();
    /// let state = state.time(Some(instant::Duration::new(0, 12)));
    /// # assert_eq!(state.time.unwrap(), instant::Duration::new(0, 12));
    /// ```
    fn time(&mut self, time: Option<instant::Duration>) -> &mut Self {
        self.time = time;
        self
    }

    /// Returns current cost function value.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State, ArgminFloat};
    /// # let mut state: IterState<Vec<f64>, (), (), (), f64> = IterState::new();
    /// # state.cost = 12.0;
    /// let cost = state.get_cost();
    /// # assert_eq!(cost.to_ne_bytes(), 12.0f64.to_ne_bytes());
    /// ```
    fn get_cost(&self) -> Self::Float {
        self.cost
    }

    /// Returns current best cost function value.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State, ArgminFloat};
    /// # let mut state: IterState<Vec<f64>, (), (), (), f64> = IterState::new();
    /// # state.best_cost = 12.0;
    /// let best_cost = state.get_best_cost();
    /// # assert_eq!(best_cost.to_ne_bytes(), 12.0f64.to_ne_bytes());
    /// ```
    fn get_best_cost(&self) -> Self::Float {
        self.best_cost
    }

    /// Returns target cost function value.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State, ArgminFloat};
    /// # let mut state: IterState<Vec<f64>, (), (), (), f64> = IterState::new();
    /// # state.target_cost = 12.0;
    /// let target_cost = state.get_target_cost();
    /// # assert_eq!(target_cost.to_ne_bytes(), 12.0f64.to_ne_bytes());
    /// ```
    fn get_target_cost(&self) -> Self::Float {
        self.target_cost
    }

    /// Returns current number of iterations.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State, ArgminFloat};
    /// # let mut state: IterState<Vec<f64>, (), (), (), f64> = IterState::new();
    /// # state.iter = 12;
    /// let iter = state.get_iter();
    /// # assert_eq!(iter, 12);
    /// ```
    fn get_iter(&self) -> u64 {
        self.iter
    }

    /// Returns iteration number of last best parameter vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State, ArgminFloat};
    /// # let mut state: IterState<Vec<f64>, (), (), (), f64> = IterState::new();
    /// # state.last_best_iter = 12;
    /// let last_best_iter = state.get_last_best_iter();
    /// # assert_eq!(last_best_iter, 12);
    /// ```
    fn get_last_best_iter(&self) -> u64 {
        self.last_best_iter
    }

    /// Returns the maximum number of iterations.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State, ArgminFloat};
    /// # let mut state: IterState<Vec<f64>, (), (), (), f64> = IterState::new();
    /// # state.max_iters = 12;
    /// let max_iters = state.get_max_iters();
    /// # assert_eq!(max_iters, 12);
    /// ```
    fn get_max_iters(&self) -> u64 {
        self.max_iters
    }

    /// Returns the termination reason.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State, ArgminFloat, TerminationReason};
    /// # let mut state: IterState<Vec<f64>, (), (), (), f64> = IterState::new();
    /// let termination_reason = state.get_termination_reason();
    /// # assert_eq!(termination_reason, TerminationReason::NotTerminated);
    /// ```
    fn get_termination_reason(&self) -> TerminationReason {
        self.termination_reason
    }

    /// Returns the time elapsed since the start of the optimization.
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate instant;
    /// # use instant;
    /// # use argmin::core::{IterState, State, ArgminFloat};
    /// # let mut state: IterState<Vec<f64>, (), (), (), f64> = IterState::new();
    /// let time = state.get_time();
    /// # assert_eq!(time.unwrap(), instant::Duration::new(0, 0));
    /// ```
    fn get_time(&self) -> Option<instant::Duration> {
        self.time
    }

    /// Increments the number of iterations by one
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State, ArgminFloat};
    /// # let mut state: IterState<Vec<f64>, (), (), (), f64> = IterState::new();
    /// # assert_eq!(state.iter, 0);
    /// state.increment_iter();
    /// # assert_eq!(state.iter, 1);
    /// ```
    fn increment_iter(&mut self) {
        self.iter += 1;
    }

    /// Set all function evaluation counts to the evaluation counts of another `Problem`.
    ///
    /// ```
    /// # use std::collections::HashMap;
    /// # use argmin::core::{Problem, IterState, State, ArgminFloat};
    /// # let mut state: IterState<Vec<f64>, (), (), (), f64> = IterState::new();
    /// # assert_eq!(state.counts, HashMap::new());
    /// # state.counts.insert("test2".to_string(), 10u64);
    /// #
    /// # #[derive(Eq, PartialEq, Debug)]
    /// # struct UserDefinedProblem {};
    /// #
    /// # let mut problem = Problem::new(UserDefinedProblem {});
    /// # problem.counts.insert("test1", 10u64);
    /// # problem.counts.insert("test2", 2);
    /// state.func_counts(&problem);
    /// # let mut hm = HashMap::new();
    /// # hm.insert("test1".to_string(), 10u64);
    /// # hm.insert("test2".to_string(), 2u64);
    /// # assert_eq!(state.counts, hm);
    /// ```
    fn func_counts<O>(&mut self, problem: &Problem<O>) {
        for (k, &v) in problem.counts.iter() {
            let count = self.counts.entry(k.to_string()).or_insert(0);
            *count = v
        }
    }

    /// Returns function evaluation counts
    ///
    /// # Example
    ///
    /// ```
    /// # use std::collections::HashMap;
    /// # use argmin::core::{IterState, State, ArgminFloat};
    /// # let mut state: IterState<Vec<f64>, (), (), (), f64> = IterState::new();
    /// # assert_eq!(state.counts, HashMap::new());
    /// # state.counts.insert("test2".to_string(), 10u64);
    /// let counts = state.get_func_counts();
    /// # let mut hm = HashMap::new();
    /// # hm.insert("test2".to_string(), 10u64);
    /// # assert_eq!(*counts, hm);
    /// ```
    fn get_func_counts(&self) -> &HashMap<String, u64> {
        &self.counts
    }

    /// Returns whether the current parameter vector is also the best parameter vector found so
    /// far.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{IterState, State, ArgminFloat};
    /// # let mut state: IterState<Vec<f64>, (), (), (), f64> = IterState::new();
    /// # state.last_best_iter = 12;
    /// # state.iter = 12;
    /// let is_best = state.is_best();
    /// # assert!(is_best);
    /// # state.last_best_iter = 12;
    /// # state.iter = 21;
    /// # let is_best = state.is_best();
    /// # assert!(!is_best);
    /// ```
    fn is_best(&self) -> bool {
        self.last_best_iter == self.iter
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::type_complexity)]
    fn test_iterstate() {
        let param = vec![1.0f64, 2.0];
        let cost: f64 = 42.0;

        let mut state: IterState<Vec<f64>, Vec<f64>, Vec<f64>, Vec<Vec<f64>>, f64> =
            IterState::new();

        assert!(state.get_param().is_none());
        assert!(state.get_prev_param().is_none());
        assert!(state.get_best_param().is_none());
        assert!(state.get_prev_best_param().is_none());

        state = state.param(param.clone());

        assert_eq!(*state.get_param().unwrap(), param);
        assert!(state.get_prev_param().is_none());
        assert!(state.get_best_param().is_none());
        assert!(state.get_prev_best_param().is_none());

        assert!(state.get_cost().is_infinite());
        assert!(state.get_cost().is_sign_positive());

        assert!(state.get_prev_cost().is_infinite());
        assert!(state.get_prev_cost().is_sign_positive());

        assert!(state.get_best_cost().is_infinite());
        assert!(state.get_best_cost().is_sign_positive());

        assert!(state.get_prev_best_cost().is_infinite());
        assert!(state.get_prev_best_cost().is_sign_positive());

        assert!(state.get_target_cost().is_infinite());
        assert!(state.get_target_cost().is_sign_negative());

        assert!(state.get_gradient().is_none());
        assert!(state.get_prev_gradient().is_none());
        assert!(state.get_hessian().is_none());
        assert!(state.get_prev_hessian().is_none());
        assert!(state.get_inv_hessian().is_none());
        assert!(state.get_prev_inv_hessian().is_none());
        assert!(state.get_jacobian().is_none());
        assert!(state.get_prev_jacobian().is_none());
        assert_eq!(state.get_iter(), 0);

        assert!(state.is_best());

        assert_eq!(state.get_max_iters(), std::u64::MAX);
        let func_counts = state.get_func_counts().clone();
        assert!(!func_counts.contains_key("cost_count"));
        assert!(!func_counts.contains_key("operator_count"));
        assert!(!func_counts.contains_key("gradient_count"));
        assert!(!func_counts.contains_key("hessian_count"));
        assert!(!func_counts.contains_key("jacobian_count"));
        assert!(!func_counts.contains_key("modify_count"));

        state = state.max_iters(42);

        assert_eq!(state.get_max_iters(), 42);

        let mut state = state.cost(cost);

        assert_eq!(state.get_cost().to_ne_bytes(), cost.to_ne_bytes());
        assert!(state.get_prev_cost().is_infinite());
        assert!(state.get_prev_cost().is_sign_positive());

        let new_param = vec![2.0, 1.0];

        state = state.param(new_param.clone());

        assert_eq!(*state.get_param().unwrap(), new_param);
        assert_eq!(*state.get_prev_param().unwrap(), param);

        let new_cost: f64 = 21.0;

        let mut state = state.cost(new_cost);

        assert_eq!(state.get_cost().to_ne_bytes(), new_cost.to_ne_bytes());
        assert_eq!(state.get_prev_cost().to_ne_bytes(), cost.to_ne_bytes());

        state.increment_iter();

        assert_eq!(state.get_iter(), 1);

        assert!(!state.is_best());

        state.last_best_iter = state.iter;

        assert!(state.is_best());

        let grad = vec![1.0, 2.0];

        let state = state.gradient(grad.clone());
        assert_eq!(*state.get_gradient().unwrap(), grad);
        assert!(state.get_prev_gradient().is_none());

        let new_grad = vec![2.0, 1.0];

        let state = state.gradient(new_grad.clone());

        assert_eq!(*state.get_gradient().unwrap(), new_grad);
        assert_eq!(*state.get_prev_gradient().unwrap(), grad);

        let hessian = vec![vec![1.0, 2.0], vec![2.0, 1.0]];

        let state = state.hessian(hessian.clone());
        assert_eq!(*state.get_hessian().unwrap(), hessian);
        assert!(state.get_prev_hessian().is_none());

        let new_hessian = vec![vec![2.0, 1.0], vec![1.0, 2.0]];

        let state = state.hessian(new_hessian.clone());

        assert_eq!(*state.get_hessian().unwrap(), new_hessian);
        assert_eq!(*state.get_prev_hessian().unwrap(), hessian);

        let inv_hessian = vec![vec![2.0, 1.0], vec![1.0, 2.0]];

        let state = state.inv_hessian(inv_hessian.clone());
        assert_eq!(*state.get_inv_hessian().unwrap(), inv_hessian);
        assert!(state.get_prev_inv_hessian().is_none());

        let new_inv_hessian = vec![vec![3.0, 4.0], vec![4.0, 3.0]];

        let state = state.inv_hessian(new_inv_hessian.clone());

        assert_eq!(*state.get_inv_hessian().unwrap(), new_inv_hessian);
        assert_eq!(*state.get_prev_inv_hessian().unwrap(), inv_hessian);

        let jacobian = vec![1.0f64, 2.0];

        let state = state.jacobian(jacobian.clone());
        assert!(state.get_prev_jacobian().is_none());

        let new_jacobian = vec![2.0f64, 1.0];

        let mut state = state.jacobian(new_jacobian.clone());

        assert_eq!(*state.get_jacobian().unwrap(), new_jacobian);
        assert_eq!(*state.get_prev_jacobian().unwrap(), jacobian);

        state.increment_iter();

        assert_eq!(state.get_iter(), 2);
        assert_eq!(state.get_last_best_iter(), 1);
        assert!(!state.is_best());

        // check again!
        assert_eq!(state.get_iter(), 2);
        assert_eq!(state.get_last_best_iter(), 1);
        assert_eq!(state.get_max_iters(), 42);

        assert!(!state.is_best());

        assert_eq!(state.get_cost().to_ne_bytes(), new_cost.to_ne_bytes());
        assert_eq!(state.get_prev_cost().to_ne_bytes(), cost.to_ne_bytes());
        assert_eq!(state.get_prev_cost().to_ne_bytes(), cost.to_ne_bytes());

        assert_eq!(*state.get_param().unwrap(), new_param);
        assert_eq!(*state.get_prev_param().unwrap(), param);

        assert_eq!(*state.get_gradient().unwrap(), new_grad);
        assert_eq!(*state.get_prev_gradient().unwrap(), grad);
        assert_eq!(*state.get_hessian().unwrap(), new_hessian);
        assert_eq!(*state.get_prev_hessian().unwrap(), hessian);
        assert_eq!(*state.get_inv_hessian().unwrap(), new_inv_hessian);
        assert_eq!(*state.get_prev_inv_hessian().unwrap(), inv_hessian);
        assert_eq!(*state.get_jacobian().unwrap(), new_jacobian);
        assert_eq!(*state.get_prev_jacobian().unwrap(), jacobian);
        assert_eq!(state.take_gradient().unwrap(), new_grad);
        assert_eq!(state.take_prev_gradient().unwrap(), grad);
        assert_eq!(state.take_hessian().unwrap(), new_hessian);
        assert_eq!(state.take_prev_hessian().unwrap(), hessian);
        assert_eq!(state.take_inv_hessian().unwrap(), new_inv_hessian);
        assert_eq!(state.take_prev_inv_hessian().unwrap(), inv_hessian);
        assert_eq!(state.take_jacobian().unwrap(), new_jacobian);
        assert_eq!(state.take_prev_jacobian().unwrap(), jacobian);
        let func_counts = state.get_func_counts().clone();
        assert!(!func_counts.contains_key("cost_count"));
        assert!(!func_counts.contains_key("operator_count"));
        assert!(!func_counts.contains_key("gradient_count"));
        assert!(!func_counts.contains_key("hessian_count"));
        assert!(!func_counts.contains_key("jacobian_count"));
        assert!(!func_counts.contains_key("modify_count"));
    }
}
