// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::{ArgminOp, OpWrapper, TerminationReason};
use instant;
use num::traits::float::Float;
use paste::item;
use serde::{Deserialize, Serialize};

/// Maintains the state from iteration to iteration of a solver
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IterState<O: ArgminOp> {
    /// Current parameter vector
    pub param: O::Param,
    /// Previous parameter vector
    pub prev_param: O::Param,
    /// Current best parameter vector
    pub best_param: O::Param,
    /// Previous best parameter vector
    pub prev_best_param: O::Param,
    /// Current cost function value
    pub cost: O::Float,
    /// Previous cost function value
    pub prev_cost: O::Float,
    /// Current best cost function value
    pub best_cost: O::Float,
    /// Previous best cost function value
    pub prev_best_cost: O::Float,
    /// Target cost function value
    pub target_cost: O::Float,
    /// Current gradient
    pub grad: Option<O::Param>,
    /// Previous gradient
    pub prev_grad: Option<O::Param>,
    /// Current Hessian
    pub hessian: Option<O::Hessian>,
    /// Previous Hessian
    pub prev_hessian: Option<O::Hessian>,
    /// Current Jacobian
    pub jacobian: Option<O::Jacobian>,
    /// Previous Jacobian
    pub prev_jacobian: Option<O::Jacobian>,
    /// All members for population-based algorithms as (param, cost) tuples
    pub population: Option<Vec<(O::Param, O::Float)>>,
    /// Current iteration
    pub iter: u64,
    /// Iteration number of last best cost
    pub last_best_iter: u64,
    /// Maximum number of iterations
    pub max_iters: u64,
    /// Number of cost function evaluations so far
    pub cost_func_count: u64,
    /// Number of gradient evaluations so far
    pub grad_func_count: u64,
    /// Number of Hessian evaluations so far
    pub hessian_func_count: u64,
    /// Number of Jacobian evaluations so far
    pub jacobian_func_count: u64,
    /// Number of modify evaluations so far
    pub modify_func_count: u64,
    /// Time required so far
    pub time: Option<instant::Duration>,
    /// Reason of termination
    pub termination_reason: TerminationReason,
}

macro_rules! setter {
    ($name:ident, $type:ty, $doc:tt) => {
        #[doc=$doc]
        pub fn $name(&mut self, $name: $type) -> &mut Self {
            self.$name = $name;
            self
        }
    };
}

macro_rules! getter_option {
    ($name:ident, $type:ty, $doc:tt) => {
        item! {
            #[doc=$doc]
            pub fn [<get_ $name>](&self) -> Option<$type> {
                self.$name.clone()
            }
        }
    };
}

macro_rules! getter {
    ($name:ident, $type:ty, $doc:tt) => {
        item! {
            #[doc=$doc]
            pub fn [<get_ $name>](&self) -> $type {
                self.$name.clone()
            }
        }
    };
}

impl<O: ArgminOp> std::default::Default for IterState<O>
where
    O::Param: Default,
{
    fn default() -> Self {
        IterState::new(O::Param::default())
    }
}

impl<O: ArgminOp> IterState<O> {
    /// Create new IterState from `param`
    pub fn new(param: O::Param) -> Self {
        IterState {
            param: param.clone(),
            prev_param: param.clone(),
            best_param: param.clone(),
            prev_best_param: param,
            cost: O::Float::infinity(),
            prev_cost: O::Float::infinity(),
            best_cost: O::Float::infinity(),
            prev_best_cost: O::Float::infinity(),
            target_cost: O::Float::neg_infinity(),
            grad: None,
            prev_grad: None,
            hessian: None,
            prev_hessian: None,
            jacobian: None,
            prev_jacobian: None,
            population: None,
            iter: 0,
            last_best_iter: 0,
            max_iters: std::u64::MAX,
            cost_func_count: 0,
            grad_func_count: 0,
            hessian_func_count: 0,
            jacobian_func_count: 0,
            modify_func_count: 0,
            time: Some(instant::Duration::new(0, 0)),
            termination_reason: TerminationReason::NotTerminated,
        }
    }

    /// Set parameter vector. This shifts the stored parameter vector to the previous parameter
    /// vector.
    pub fn param(&mut self, param: O::Param) -> &mut Self {
        std::mem::swap(&mut self.prev_param, &mut self.param);
        self.param = param;
        self
    }

    /// Set best paramater vector. This shifts the stored best parameter vector to the previous
    /// best parameter vector.
    pub fn best_param(&mut self, param: O::Param) -> &mut Self {
        std::mem::swap(&mut self.prev_best_param, &mut self.best_param);
        self.best_param = param;
        self
    }

    /// Set the current cost function value. This shifts the stored cost function value to the
    /// previous cost function value.
    pub fn cost(&mut self, cost: O::Float) -> &mut Self {
        std::mem::swap(&mut self.prev_cost, &mut self.cost);
        self.cost = cost;
        self
    }

    /// Set the current best cost function value. This shifts the stored best cost function value to
    /// the previous cost function value.
    pub fn best_cost(&mut self, cost: O::Float) -> &mut Self {
        std::mem::swap(&mut self.prev_best_cost, &mut self.best_cost);
        self.best_cost = cost;
        self
    }

    /// Set gradient. This shifts the stored gradient to the previous gradient.
    pub fn grad(&mut self, grad: O::Param) -> &mut Self {
        std::mem::swap(&mut self.prev_grad, &mut self.grad);
        self.grad = Some(grad);
        self
    }

    /// Set Hessian. This shifts the stored Hessian to the previous Hessian.
    pub fn hessian(&mut self, hessian: O::Hessian) -> &mut Self {
        std::mem::swap(&mut self.prev_hessian, &mut self.hessian);
        self.hessian = Some(hessian);
        self
    }

    /// Set Jacobian. This shifts the stored Jacobian to the previous Jacobian.
    pub fn jacobian(&mut self, jacobian: O::Jacobian) -> &mut Self {
        std::mem::swap(&mut self.prev_jacobian, &mut self.jacobian);
        self.jacobian = Some(jacobian);
        self
    }

    /// Set population
    pub fn population(&mut self, population: Vec<(O::Param, O::Float)>) -> &mut Self {
        self.population = Some(population);
        self
    }

    setter!(target_cost, O::Float, "Set target cost value");
    setter!(max_iters, u64, "Set maximum number of iterations");
    setter!(
        last_best_iter,
        u64,
        "Set iteration number where the previous best parameter vector was found"
    );
    setter!(
        termination_reason,
        TerminationReason,
        "Set termination_reason"
    );
    setter!(time, Option<instant::Duration>, "Set time required so far");
    getter!(param, O::Param, "Returns current parameter vector");
    getter!(prev_param, O::Param, "Returns previous parameter vector");
    getter!(best_param, O::Param, "Returns best parameter vector");
    getter!(
        prev_best_param,
        O::Param,
        "Returns previous best parameter vector"
    );
    getter!(cost, O::Float, "Returns current cost function value");
    getter!(prev_cost, O::Float, "Returns previous cost function value");
    getter!(
        best_cost,
        O::Float,
        "Returns current best cost function value"
    );
    getter!(
        prev_best_cost,
        O::Float,
        "Returns previous best cost function value"
    );
    getter!(target_cost, O::Float, "Returns target cost");
    getter!(
        cost_func_count,
        u64,
        "Returns current cost function evaluation count"
    );
    getter!(
        grad_func_count,
        u64,
        "Returns current gradient function evaluation count"
    );
    getter!(
        hessian_func_count,
        u64,
        "Returns current Hessian function evaluation count"
    );
    getter!(
        jacobian_func_count,
        u64,
        "Returns current Jacobian function evaluation count"
    );
    getter!(
        modify_func_count,
        u64,
        "Returns current Modify function evaluation count"
    );
    getter!(
        last_best_iter,
        u64,
        "Returns iteration number where the last best parameter vector was found"
    );
    getter!(
        termination_reason,
        TerminationReason,
        "Get termination_reason"
    );
    getter!(time, Option<instant::Duration>, "Get time required so far");
    getter_option!(grad, O::Param, "Returns gradient");
    getter_option!(prev_grad, O::Param, "Returns previous gradient");
    getter_option!(hessian, O::Hessian, "Returns current Hessian");
    getter_option!(prev_hessian, O::Hessian, "Returns previous Hessian");
    getter_option!(jacobian, O::Jacobian, "Returns current Jacobian");
    getter_option!(prev_jacobian, O::Jacobian, "Returns previous Jacobian");
    getter!(iter, u64, "Returns current number of iterations");
    getter!(max_iters, u64, "Returns maximum number of iterations");

    /// Returns population
    pub fn get_population(&self) -> Option<&Vec<(O::Param, O::Float)>> {
        self.population.as_ref()
    }

    /// Increment the number of iterations by one
    pub fn increment_iter(&mut self) {
        self.iter += 1;
    }

    /// Increment all function evaluation counts by the evaluation counts of another operator
    /// wrapped in `OpWrapper`.
    pub fn increment_func_counts(&mut self, op: &OpWrapper<O>) {
        self.cost_func_count += op.cost_func_count;
        self.grad_func_count += op.grad_func_count;
        self.hessian_func_count += op.hessian_func_count;
        self.jacobian_func_count += op.jacobian_func_count;
        self.modify_func_count += op.modify_func_count;
    }

    /// Set all function evaluation counts to the evaluation counts of another operator
    /// wrapped in `OpWrapper`.
    pub fn set_func_counts(&mut self, op: &OpWrapper<O>) {
        self.cost_func_count = op.cost_func_count;
        self.grad_func_count = op.grad_func_count;
        self.hessian_func_count = op.hessian_func_count;
        self.jacobian_func_count = op.jacobian_func_count;
        self.modify_func_count = op.modify_func_count;
    }

    /// Increment cost function evaluation count by `num`
    pub fn increment_cost_func_count(&mut self, num: u64) {
        self.cost_func_count += num;
    }

    /// Increment gradient function evaluation count by `num`
    pub fn increment_grad_func_count(&mut self, num: u64) {
        self.grad_func_count += num;
    }

    /// Increment Hessian function evaluation count by `num`
    pub fn increment_hessian_func_count(&mut self, num: u64) {
        self.hessian_func_count += num;
    }

    /// Increment Jacobian function evaluation count by `num`
    pub fn increment_jacobian_func_count(&mut self, num: u64) {
        self.jacobian_func_count += num;
    }

    /// Increment modify function evaluation count by `num`
    pub fn increment_modify_func_count(&mut self, num: u64) {
        self.modify_func_count += num;
    }

    /// Indicate that a new best parameter vector was found
    pub fn new_best(&mut self) {
        self.last_best_iter = self.iter;
    }

    /// Returns whether the current parameter vector is also the best parameter vector found so
    /// far.
    pub fn is_best(&self) -> bool {
        self.last_best_iter == self.iter
    }

    /// Return whether the algorithm has terminated or not
    pub fn terminated(&self) -> bool {
        self.termination_reason.terminated()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::MinimalNoOperator;

    #[test]
    fn test_iterstate() {
        let param = vec![1.0f64, 2.0];
        let cost: f64 = 42.0;

        let mut state: IterState<MinimalNoOperator> = IterState::new(param.clone());

        assert_eq!(state.get_param(), param);
        assert_eq!(state.get_prev_param(), param);
        assert_eq!(state.get_best_param(), param);
        assert_eq!(state.get_prev_best_param(), param);
        assert_eq!(state.get_cost(), std::f64::INFINITY);
        assert_eq!(state.get_prev_cost(), std::f64::INFINITY);
        assert_eq!(state.get_best_cost(), std::f64::INFINITY);
        assert_eq!(state.get_prev_best_cost(), std::f64::INFINITY);
        assert_eq!(state.get_target_cost(), std::f64::NEG_INFINITY);
        assert_eq!(state.get_grad(), None);
        assert_eq!(state.get_prev_grad(), None);
        assert_eq!(state.get_hessian(), None);
        assert_eq!(state.get_prev_hessian(), None);
        assert_eq!(state.get_jacobian(), None);
        assert_eq!(state.get_prev_jacobian(), None);
        assert_eq!(state.get_iter(), 0);
        assert_eq!(state.is_best(), true);
        assert_eq!(state.get_max_iters(), std::u64::MAX);
        assert_eq!(state.get_cost_func_count(), 0);
        assert_eq!(state.get_grad_func_count(), 0);
        assert_eq!(state.get_hessian_func_count(), 0);
        assert_eq!(state.get_jacobian_func_count(), 0);
        assert_eq!(state.get_modify_func_count(), 0);

        state.max_iters(42);

        assert_eq!(state.get_max_iters(), 42);

        state.cost(cost);

        assert_eq!(state.get_cost(), cost);
        assert_eq!(state.get_prev_cost(), std::f64::INFINITY);

        state.best_cost(cost);

        assert_eq!(state.get_best_cost(), cost);
        assert_eq!(state.get_prev_best_cost(), std::f64::INFINITY);

        let new_param = vec![2.0, 1.0];

        state.param(new_param.clone());

        assert_eq!(state.get_param(), new_param);
        assert_eq!(state.get_prev_param(), param);

        state.best_param(new_param.clone());

        assert_eq!(state.get_best_param(), new_param);
        assert_eq!(state.get_prev_best_param(), param);

        let new_cost = 21.0;

        state.cost(new_cost);

        assert_eq!(state.get_cost(), new_cost);
        assert_eq!(state.get_prev_cost(), cost);

        state.best_cost(new_cost);

        assert_eq!(state.get_best_cost(), new_cost);
        assert_eq!(state.get_prev_best_cost(), cost);

        state.increment_iter();

        assert_eq!(state.get_iter(), 1);

        assert_eq!(state.is_best(), false);

        state.new_best();

        assert_eq!(state.is_best(), true);

        let grad = vec![1.0, 2.0];

        state.grad(grad.clone());
        assert_eq!(state.get_grad(), Some(grad.clone()));
        assert_eq!(state.get_prev_grad(), None);

        let new_grad = vec![2.0, 1.0];

        state.grad(new_grad.clone());

        assert_eq!(state.get_grad(), Some(new_grad.clone()));
        assert_eq!(state.get_prev_grad(), Some(grad.clone()));

        let hessian = vec![vec![1.0, 2.0], vec![2.0, 1.0]];

        state.hessian(hessian.clone());
        assert_eq!(state.get_hessian(), Some(hessian.clone()));
        assert_eq!(state.get_prev_hessian(), None);

        let new_hessian = vec![vec![2.0, 1.0], vec![1.0, 2.0]];

        state.hessian(new_hessian.clone());

        assert_eq!(state.get_hessian(), Some(new_hessian.clone()));
        assert_eq!(state.get_prev_hessian(), Some(hessian.clone()));

        let jacobian = vec![1.0, 2.0];

        state.jacobian(jacobian.clone());
        assert_eq!(state.get_jacobian(), Some(jacobian.clone()));
        assert_eq!(state.get_prev_jacobian(), None);

        let new_jacobian = vec![2.0, 1.0];

        state.jacobian(new_jacobian.clone());

        assert_eq!(state.get_jacobian(), Some(new_jacobian.clone()));
        assert_eq!(state.get_prev_jacobian(), Some(jacobian.clone()));

        state.increment_iter();

        assert_eq!(state.get_iter(), 2);
        assert_eq!(state.get_last_best_iter(), 1);
        assert_eq!(state.is_best(), false);

        state.increment_cost_func_count(42);
        assert_eq!(state.get_cost_func_count(), 42);
        state.increment_grad_func_count(43);
        assert_eq!(state.get_grad_func_count(), 43);
        state.increment_hessian_func_count(44);
        assert_eq!(state.get_hessian_func_count(), 44);
        state.increment_jacobian_func_count(46);
        assert_eq!(state.get_jacobian_func_count(), 46);
        state.increment_modify_func_count(45);
        assert_eq!(state.get_modify_func_count(), 45);

        // check again!
        assert_eq!(state.get_iter(), 2);
        assert_eq!(state.get_last_best_iter(), 1);
        assert_eq!(state.get_max_iters(), 42);
        assert_eq!(state.is_best(), false);
        assert_eq!(state.get_cost(), new_cost);
        assert_eq!(state.get_prev_cost(), cost);
        assert_eq!(state.get_param(), new_param);
        assert_eq!(state.get_prev_param(), param);
        assert_eq!(state.get_best_cost(), new_cost);
        assert_eq!(state.get_prev_best_cost(), cost);
        assert_eq!(state.get_best_param(), new_param);
        assert_eq!(state.get_prev_best_param(), param);
        assert_eq!(state.get_best_cost(), new_cost);
        assert_eq!(state.get_prev_best_cost(), cost);
        assert_eq!(state.get_grad(), Some(new_grad));
        assert_eq!(state.get_prev_grad(), Some(grad));
        assert_eq!(state.get_hessian(), Some(new_hessian));
        assert_eq!(state.get_prev_hessian(), Some(hessian));
        assert_eq!(state.get_jacobian(), Some(new_jacobian));
        assert_eq!(state.get_prev_jacobian(), Some(jacobian));
        assert_eq!(state.get_cost_func_count(), 42);
        assert_eq!(state.get_grad_func_count(), 43);
        assert_eq!(state.get_hessian_func_count(), 44);
        assert_eq!(state.get_jacobian_func_count(), 46);
        assert_eq!(state.get_modify_func_count(), 45);
    }
}
