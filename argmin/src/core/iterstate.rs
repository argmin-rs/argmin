// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::{
    ArgminFloat, ArgminIterData, ArgminOp, DeserializeOwnedAlias, LinearProgram, OpWrapper,
    SerializeAlias, TerminationReason,
};
use instant;
use num_traits::Float;
use paste::item;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// Types implemeting this trait can be used to keep track of a solver's state
pub trait State: SerializeAlias + Sized {
    /// Parameter vector
    type Param: Clone + SerializeAlias + DeserializeOwnedAlias;
    /// Output of the operator
    type Output: Clone + SerializeAlias + DeserializeOwnedAlias;
    /// Type of Hessian
    type Hessian: Clone + SerializeAlias + DeserializeOwnedAlias;
    /// Type of Jacobian
    type Jacobian: Clone + SerializeAlias + DeserializeOwnedAlias;
    /// Precision of floats
    type Float: ArgminFloat;
    /// Operator
    type Operator;

    /// Constructor
    fn new(param: Self::Param) -> Self;

    /// Update stored data with information from an `ArgminIterData` struct
    fn update(&mut self, data: &ArgminIterData<Self>);

    /// Set the current parameter as the new best parameter
    fn current_param_is_new_best(&mut self);

    /// Set parameter vector. This shifts the stored parameter vector to the previous parameter
    /// vector.
    fn param(&mut self, _param: Self::Param) -> &mut Self {
        unimplemented!()
    }

    /// Set best paramater vector. This shifts the stored best parameter vector to the previous
    /// best parameter vector.
    fn best_param(&mut self, _param: Self::Param) -> &mut Self {
        unimplemented!()
    }

    /// Set the current cost function value. This shifts the stored cost function value to the
    /// previous cost function value.
    fn cost(&mut self, _cost: Self::Float) -> &mut Self {
        unimplemented!()
    }

    /// Set the current best cost function value. This shifts the stored best cost function value to
    /// the previous cost function value.
    fn best_cost(&mut self, _cost: Self::Float) -> &mut Self {
        unimplemented!()
    }

    /// Set gradient. This shifts the stored gradient to the previous gradient.
    fn grad(&mut self, _grad: Self::Param) -> &mut Self {
        unimplemented!()
    }

    /// Set Hessian. This shifts the stored Hessian to the previous Hessian.
    fn hessian(&mut self, _hessian: Self::Hessian) -> &mut Self {
        unimplemented!()
    }

    /// Set inverse Hessian. This shifts the stored inverse Hessian to the previous inverse Hessian.
    fn inv_hessian(&mut self, _inv_hessian: Self::Hessian) -> &mut Self {
        unimplemented!()
    }

    /// Set Jacobian. This shifts the stored Jacobian to the previous Jacobian.
    fn jacobian(&mut self, _jacobian: Self::Jacobian) -> &mut Self {
        unimplemented!()
    }

    /// Set population
    fn population(&mut self, _population: Vec<(Self::Param, Self::Float)>) -> &mut Self {
        unimplemented!()
    }

    /// Set target cost
    fn target_cost(&mut self, _target_cost: Self::Float) -> &mut Self {
        unimplemented!()
    }

    /// Set maximum number of iterations
    fn max_iters(&mut self, _max_iters: u64) -> &mut Self {
        unimplemented!()
    }

    /// Set last best iteration
    fn last_best_iter(&mut self, _last_best_iter: u64) -> &mut Self {
        unimplemented!()
    }

    /// Set termination reason
    fn termination_reason(&mut self, _termination_reason: TerminationReason) -> &mut Self {
        unimplemented!()
    }

    /// Set time required so far
    fn time(&mut self, _time: Option<instant::Duration>) -> &mut Self {
        unimplemented!()
    }

    /// Returns current parameter vector
    fn get_param(&self) -> Option<Self::Param> {
        unimplemented!()
    }

    /// Returns previous parameter vector
    fn get_prev_param(&self) -> Option<Self::Param> {
        unimplemented!()
    }

    /// Returns best parameter vector
    fn get_best_param(&self) -> Option<Self::Param> {
        unimplemented!()
    }

    /// Returns previous best parameter vector
    fn get_prev_best_param(&self) -> Option<Self::Param> {
        unimplemented!()
    }

    /// Returns reference to current parameter vector
    fn get_param_ref(&self) -> Option<&Self::Param> {
        unimplemented!()
    }

    /// Returns reference to previous parameter vector
    fn get_prev_param_ref(&self) -> Option<&Self::Param> {
        unimplemented!()
    }

    /// Returns reference to best parameter vector
    fn get_best_param_ref(&self) -> Option<&Self::Param> {
        unimplemented!()
    }

    /// Returns reference to previous best parameter vector
    fn get_prev_best_param_ref(&self) -> Option<&Self::Param> {
        unimplemented!()
    }

    /// Takes current parameter vector and resplaces it internally with `None`
    fn take_param(&mut self) -> Option<Self::Param> {
        unimplemented!()
    }

    /// Takes previous parameter vector and resplaces it internally with `None`
    fn take_prev_param(&mut self) -> Option<Self::Param> {
        unimplemented!()
    }

    /// Takes best parameter vector and resplaces it internally with `None`
    fn take_best_param(&mut self) -> Option<Self::Param> {
        unimplemented!()
    }

    /// Takes previous best parameter vector and resplaces it internally with `None`
    fn take_prev_best_param(&mut self) -> Option<Self::Param> {
        unimplemented!()
    }

    /// Returns current cost function value
    fn get_cost(&self) -> Self::Float {
        unimplemented!()
    }

    /// Returns previous cost function value
    fn get_prev_cost(&self) -> Self::Float {
        unimplemented!()
    }

    /// Returns best cost function value
    fn get_best_cost(&self) -> Self::Float {
        unimplemented!()
    }

    /// Returns previous best cost function value
    fn get_prev_best_cost(&self) -> Self::Float {
        unimplemented!()
    }

    /// Returns target cost
    fn get_target_cost(&self) -> Self::Float {
        unimplemented!()
    }

    /// Returns currecnt cost function evaluation count
    fn get_cost_func_count(&self) -> u64 {
        unimplemented!()
    }

    /// Returns current gradient function evaluation count
    fn get_grad_func_count(&self) -> u64 {
        unimplemented!()
    }

    /// Returns current Hessian function evaluation count
    fn get_hessian_func_count(&self) -> u64 {
        unimplemented!()
    }

    /// Returns current Jacobian function evaluation count
    fn get_jacobian_func_count(&self) -> u64 {
        unimplemented!()
    }

    /// Returns current modify function evaluation count
    fn get_modify_func_count(&self) -> u64 {
        unimplemented!()
    }

    /// Returns iteration number where the last best parameter vector was found
    fn get_last_best_iter(&self) -> u64 {
        unimplemented!()
    }

    /// Returns current number of iterations
    fn get_iter(&self) -> u64 {
        unimplemented!()
    }

    /// Returns maximum number of iterations
    fn get_max_iters(&self) -> u64 {
        unimplemented!()
    }

    /// Returns termination reason
    fn get_termination_reason(&self) -> TerminationReason {
        unimplemented!()
    }

    /// Get time required so far
    fn get_time(&self) -> Option<instant::Duration> {
        unimplemented!()
    }

    /// Returns gradient
    fn get_grad(&self) -> Option<Self::Param> {
        unimplemented!()
    }

    /// Returns previous gradient
    fn get_prev_grad(&self) -> Option<Self::Param> {
        unimplemented!()
    }

    /// Returns current Hessian
    fn get_hessian(&self) -> Option<Self::Hessian> {
        unimplemented!()
    }

    /// Returns previous Hessian
    fn get_prev_hessian(&self) -> Option<Self::Hessian> {
        unimplemented!()
    }

    /// Returns current inverse Hessian
    fn get_inv_hessian(&self) -> Option<Self::Hessian> {
        unimplemented!()
    }

    /// Returns previous inverse Hessian
    fn get_prev_inv_hessian(&self) -> Option<Self::Hessian> {
        unimplemented!()
    }

    /// Returns current Jacobian
    fn get_jacobian(&self) -> Option<Self::Jacobian> {
        unimplemented!()
    }

    /// Returns previous Jacobian
    fn get_prev_jacobian(&self) -> Option<Self::Jacobian> {
        unimplemented!()
    }

    /// Returns reference to gradient
    fn get_grad_ref(&self) -> Option<&Self::Param> {
        unimplemented!()
    }

    /// Returns reference to previous gradient
    fn get_prev_grad_ref(&self) -> Option<&Self::Param> {
        unimplemented!()
    }

    /// Returns reference to current Hessian
    fn get_hessian_ref(&self) -> Option<&Self::Hessian> {
        unimplemented!()
    }

    /// Returns reference to previous Hessian
    fn get_prev_hessian_ref(&self) -> Option<&Self::Hessian> {
        unimplemented!()
    }

    /// Returns reference to current inverse Hessian
    fn get_inv_hessian_ref(&self) -> Option<&Self::Hessian> {
        unimplemented!()
    }

    /// Returns reference to previous inverse Hessian
    fn get_prev_inv_hessian_ref(&self) -> Option<&Self::Hessian> {
        unimplemented!()
    }

    /// Returns reference to current Jacobian
    fn get_jacobian_ref(&self) -> Option<&Self::Jacobian> {
        unimplemented!()
    }

    /// Returns reference to previous Jacobian
    fn get_prev_jacobian_ref(&self) -> Option<&Self::Jacobian> {
        unimplemented!()
    }

    /// Takes gradient
    fn take_grad(&mut self) -> Option<Self::Param> {
        unimplemented!()
    }

    /// Takes previous gradient
    fn take_prev_grad(&mut self) -> Option<Self::Param> {
        unimplemented!()
    }

    /// Takes current Hessian
    fn take_hessian(&mut self) -> Option<Self::Hessian> {
        unimplemented!()
    }

    /// Takes previous Hessian
    fn take_prev_hessian(&mut self) -> Option<Self::Hessian> {
        unimplemented!()
    }

    /// Takes current inverse Hessian
    fn take_inv_hessian(&mut self) -> Option<Self::Hessian> {
        unimplemented!()
    }

    /// Takes previous inverse Hessian
    fn take_prev_inv_hessian(&mut self) -> Option<Self::Hessian> {
        unimplemented!()
    }

    /// Takes current Jacobian
    fn take_jacobian(&mut self) -> Option<Self::Jacobian> {
        unimplemented!()
    }

    /// Takes previous Jacobian and replaces it internally with `None`
    fn take_prev_jacobian(&mut self) -> Option<Self::Jacobian> {
        unimplemented!()
    }

    /// Returns population
    fn get_population(&self) -> Option<&Vec<(Self::Param, Self::Float)>> {
        unimplemented!()
    }

    /// Increment the number of iterations by one
    fn increment_iter(&mut self) {
        unimplemented!()
    }

    /// Increment all function evaluation counts by the evaluation counts of another operator
    /// wrapped in `OpWrapper`.
    fn increment_func_counts(&mut self, _op: &OpWrapper<Self::Operator>) {
        unimplemented!()
    }

    /// Set all function evaluation counts to the evaluation counts of another operator
    /// wrapped in `OpWrapper`.
    fn set_func_counts(&mut self, _op: &OpWrapper<Self::Operator>) {
        unimplemented!()
    }

    /// Increment cost function evaluation count by `num`
    fn increment_cost_func_count(&mut self, _num: u64) {
        unimplemented!()
    }

    /// Increment gradient function evaluation count by `num`
    fn increment_grad_func_count(&mut self, _num: u64) {
        unimplemented!()
    }

    /// Increment Hessian function evaluation count by `num`
    fn increment_hessian_func_count(&mut self, _num: u64) {
        unimplemented!()
    }

    /// Increment Jacobian function evaluation count by `num`
    fn increment_jacobian_func_count(&mut self, _num: u64) {
        unimplemented!()
    }

    /// Increment modify function evaluation count by `num`
    fn increment_modify_func_count(&mut self, _num: u64) {
        unimplemented!()
    }

    /// Indicate that a new best parameter vector was found
    fn new_best(&mut self) {
        unimplemented!()
    }

    /// Returns whether the current parameter vector is also the best parameter vector found so
    /// far.
    fn is_best(&self) -> bool {
        unimplemented!()
    }

    /// Return whether the algorithm has terminated or not
    fn terminated(&self) -> bool {
        unimplemented!()
    }
}

/// Maintains the state from iteration to iteration of a solver
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct IterState<O: ArgminOp> {
    /// Current parameter vector
    pub param: Option<O::Param>,
    /// Previous parameter vector
    pub prev_param: Option<O::Param>,
    /// Current best parameter vector
    pub best_param: Option<O::Param>,
    /// Previous best parameter vector
    pub prev_best_param: Option<O::Param>,
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
    /// Current inverse Hessian
    pub inv_hessian: Option<O::Hessian>,
    /// Previous inverse Hessian
    pub prev_inv_hessian: Option<O::Hessian>,
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
        fn $name(&mut self, $name: $type) -> &mut Self {
            self.$name = $name;
            self
        }
    };
}

macro_rules! getter_option {
    ($name:ident, $type:ty, $doc:tt) => {
        item! {
            #[doc=$doc]
            fn [<get_ $name>](&self) -> Option<$type> {
                self.$name.clone()
            }
        }
    };
}

macro_rules! getter_option_ref {
    ($name:ident, $type:ty, $doc:tt) => {
        item! {
            #[doc=$doc]
            fn [<get_ $name _ref>](&self) -> Option<&$type> {
                self.$name.as_ref()
            }
        }
    };
}

macro_rules! take {
    ($name:ident, $type:ty, $doc:tt) => {
        item! {
            #[doc=$doc]
            fn [<take_ $name>](&mut self) -> Option<$type> {
                self.$name.take()
            }
        }
    };
}

macro_rules! getter {
    ($name:ident, $type:ty, $doc:tt) => {
        item! {
            #[doc=$doc]
            fn [<get_ $name>](&self) -> $type {
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

impl<O: ArgminOp> State for IterState<O> {
    type Param = O::Param;
    type Output = O::Output;
    type Hessian = O::Hessian;
    type Jacobian = O::Jacobian;
    type Float = O::Float;
    type Operator = O;

    /// Create new IterState from `param`
    fn new(param: O::Param) -> Self {
        IterState {
            param: Some(param.clone()),
            prev_param: None,
            best_param: Some(param),
            prev_best_param: None,
            cost: O::Float::infinity(),
            prev_cost: O::Float::infinity(),
            best_cost: O::Float::infinity(),
            prev_best_cost: O::Float::infinity(),
            target_cost: O::Float::neg_infinity(),
            grad: None,
            prev_grad: None,
            hessian: None,
            prev_hessian: None,
            inv_hessian: None,
            prev_inv_hessian: None,
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

    fn update(&mut self, data: &ArgminIterData<IterState<O>>) {
        if let Some(cur_param) = data.get_param() {
            self.param(cur_param);
        }
        if let Some(cur_cost) = data.get_cost() {
            self.cost(cur_cost);
        }
        // check if parameters are the best so far
        // Comparison is done using `<` to avoid new solutions with the same cost function value as
        // the current best to be accepted. However, some solvers to not compute the cost function
        // value (such as the Newton method). Those will always have `Inf` cost. Therefore if both
        // the new value and the previous best value are `Inf`, the solution is also accepted. Care
        // is taken that both `Inf` also have the same sign.
        if self.get_cost() < self.get_best_cost()
            || (self.get_cost().is_infinite()
                && self.get_best_cost().is_infinite()
                && self.get_cost().is_sign_positive() == self.get_best_cost().is_sign_positive())
        {
            self.current_param_is_new_best();
        }

        if let Some(grad) = data.get_grad() {
            self.grad(grad);
        }
        if let Some(hessian) = data.get_hessian() {
            self.hessian(hessian);
        }
        if let Some(inv_hessian) = data.get_inv_hessian() {
            self.inv_hessian(inv_hessian);
        }
        if let Some(jacobian) = data.get_jacobian() {
            self.jacobian(jacobian);
        }
        if let Some(population) = data.get_population() {
            self.population(population.clone());
        }

        if let Some(termination_reason) = data.get_termination_reason() {
            self.termination_reason(termination_reason);
        }
    }

    /// Sets the current parameter vector as the new best parameter vector
    fn current_param_is_new_best(&mut self) {
        let param = self.get_param().unwrap();
        let cost = self.get_cost();
        self.best_param(param).best_cost(cost);
        self.new_best();
    }

    /// Set parameter vector. This shifts the stored parameter vector to the previous parameter
    /// vector.
    fn param(&mut self, param: O::Param) -> &mut Self {
        std::mem::swap(&mut self.prev_param, &mut self.param);
        self.param = Some(param);
        self
    }

    /// Set best paramater vector. This shifts the stored best parameter vector to the previous
    /// best parameter vector.
    fn best_param(&mut self, param: O::Param) -> &mut Self {
        std::mem::swap(&mut self.prev_best_param, &mut self.best_param);
        self.best_param = Some(param);
        self
    }

    /// Set the current cost function value. This shifts the stored cost function value to the
    /// previous cost function value.
    fn cost(&mut self, cost: O::Float) -> &mut Self {
        std::mem::swap(&mut self.prev_cost, &mut self.cost);
        self.cost = cost;
        self
    }

    /// Set the current best cost function value. This shifts the stored best cost function value to
    /// the previous cost function value.
    fn best_cost(&mut self, cost: O::Float) -> &mut Self {
        std::mem::swap(&mut self.prev_best_cost, &mut self.best_cost);
        self.best_cost = cost;
        self
    }

    /// Set gradient. This shifts the stored gradient to the previous gradient.
    fn grad(&mut self, grad: O::Param) -> &mut Self {
        std::mem::swap(&mut self.prev_grad, &mut self.grad);
        self.grad = Some(grad);
        self
    }

    /// Set Hessian. This shifts the stored Hessian to the previous Hessian.
    fn hessian(&mut self, hessian: O::Hessian) -> &mut Self {
        std::mem::swap(&mut self.prev_hessian, &mut self.hessian);
        self.hessian = Some(hessian);
        self
    }

    /// Set inverse Hessian. This shifts the stored inverse Hessian to the previous inverse Hessian.
    fn inv_hessian(&mut self, inv_hessian: O::Hessian) -> &mut Self {
        std::mem::swap(&mut self.prev_inv_hessian, &mut self.inv_hessian);
        self.inv_hessian = Some(inv_hessian);
        self
    }

    /// Set Jacobian. This shifts the stored Jacobian to the previous Jacobian.
    fn jacobian(&mut self, jacobian: O::Jacobian) -> &mut Self {
        std::mem::swap(&mut self.prev_jacobian, &mut self.jacobian);
        self.jacobian = Some(jacobian);
        self
    }

    /// Set population
    fn population(&mut self, population: Vec<(O::Param, O::Float)>) -> &mut Self {
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
    getter_option!(param, O::Param, "Returns current parameter vector");
    getter_option!(prev_param, O::Param, "Returns previous parameter vector");
    getter_option_ref!(
        param,
        O::Param,
        "Returns reference to current parameter vector"
    );
    take!(
        param,
        O::Param,
        "Moves the current parameter vector out and replaces it internally with `None`"
    );
    getter_option_ref!(
        prev_param,
        O::Param,
        "Returns reference to previous parameter vector"
    );
    take!(
        prev_param,
        O::Param,
        "Moves the previous parameter vector out and replaces it internally with `None`"
    );
    getter_option!(best_param, O::Param, "Returns best parameter vector");
    getter_option!(
        prev_best_param,
        O::Param,
        "Returns previous best parameter vector"
    );
    getter_option_ref!(
        best_param,
        O::Param,
        "Returns reference to best parameter vector"
    );
    getter_option_ref!(
        prev_best_param,
        O::Param,
        "Returns reference to previous best parameter vector"
    );
    take!(
        best_param,
        O::Param,
        "Moves the best parameter vector out and replaces it internally with `None`"
    );
    take!(
        prev_best_param,
        O::Param,
        "Moves the previous best parameter vector out and replaces it internally with `None`"
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
    getter_option!(inv_hessian, O::Hessian, "Returns current inverse Hessian");
    getter_option!(
        prev_inv_hessian,
        O::Hessian,
        "Returns previous inverse Hessian"
    );
    getter_option!(jacobian, O::Jacobian, "Returns current Jacobian");
    getter_option!(prev_jacobian, O::Jacobian, "Returns previous Jacobian");
    getter_option_ref!(grad, O::Param, "Returns reference to the gradient");
    getter_option_ref!(
        prev_grad,
        O::Param,
        "Returns reference to the previous gradient"
    );
    getter_option_ref!(
        hessian,
        O::Hessian,
        "Returns reference to the current Hessian"
    );
    getter_option_ref!(
        prev_hessian,
        O::Hessian,
        "Returns reference to the previous Hessian"
    );
    getter_option_ref!(
        jacobian,
        O::Jacobian,
        "Returns reference to the current Jacobian"
    );
    getter_option_ref!(
        prev_jacobian,
        O::Jacobian,
        "Returns reference to the previous Jacobian"
    );
    getter_option_ref!(
        inv_hessian,
        O::Hessian,
        "Returns reference to the current inverse Hessian"
    );
    getter_option_ref!(
        prev_inv_hessian,
        O::Hessian,
        "Returns reference to the previous inverse Hessian"
    );
    take!(
        grad,
        O::Param,
        "Moves the gradient out and replaces it internally with `None`"
    );
    take!(
        prev_grad,
        O::Param,
        "Moves the previous gradient out and replaces it internally with `None`"
    );
    take!(
        hessian,
        O::Hessian,
        "Moves the Hessian out and replaces it internally with `None`"
    );
    take!(
        prev_hessian,
        O::Hessian,
        "Moves the previous Hessian out and replaces it internally with `None`"
    );
    take!(
        jacobian,
        O::Jacobian,
        "Moves the Jacobian out and replaces it internally with `None`"
    );
    take!(
        prev_jacobian,
        O::Jacobian,
        "Moves the previous Jacobian out and replaces it internally with `None`"
    );
    take!(
        inv_hessian,
        O::Hessian,
        "Moves the inverse Hessian out and replaces it internally with `None`"
    );
    take!(
        prev_inv_hessian,
        O::Hessian,
        "Moves the previous inverse Hessian out and replaces it internally with `None`"
    );
    getter!(iter, u64, "Returns current number of iterations");
    getter!(max_iters, u64, "Returns maximum number of iterations");

    /// Returns population
    fn get_population(&self) -> Option<&Vec<(O::Param, O::Float)>> {
        self.population.as_ref()
    }

    /// Increment the number of iterations by one
    fn increment_iter(&mut self) {
        self.iter += 1;
    }

    /// Increment all function evaluation counts by the evaluation counts of another operator
    /// wrapped in `OpWrapper`.
    fn increment_func_counts(&mut self, op: &OpWrapper<O>) {
        self.cost_func_count += op.cost_func_count;
        self.grad_func_count += op.grad_func_count;
        self.hessian_func_count += op.hessian_func_count;
        self.jacobian_func_count += op.jacobian_func_count;
        self.modify_func_count += op.modify_func_count;
    }

    /// Set all function evaluation counts to the evaluation counts of another operator
    /// wrapped in `OpWrapper`.
    fn set_func_counts(&mut self, op: &OpWrapper<O>) {
        self.cost_func_count = op.cost_func_count;
        self.grad_func_count = op.grad_func_count;
        self.hessian_func_count = op.hessian_func_count;
        self.jacobian_func_count = op.jacobian_func_count;
        self.modify_func_count = op.modify_func_count;
    }

    /// Increment cost function evaluation count by `num`
    fn increment_cost_func_count(&mut self, num: u64) {
        self.cost_func_count += num;
    }

    /// Increment gradient function evaluation count by `num`
    fn increment_grad_func_count(&mut self, num: u64) {
        self.grad_func_count += num;
    }

    /// Increment Hessian function evaluation count by `num`
    fn increment_hessian_func_count(&mut self, num: u64) {
        self.hessian_func_count += num;
    }

    /// Increment Jacobian function evaluation count by `num`
    fn increment_jacobian_func_count(&mut self, num: u64) {
        self.jacobian_func_count += num;
    }

    /// Increment modify function evaluation count by `num`
    fn increment_modify_func_count(&mut self, num: u64) {
        self.modify_func_count += num;
    }

    /// Indicate that a new best parameter vector was found
    fn new_best(&mut self) {
        self.last_best_iter = self.iter;
    }

    /// Returns whether the current parameter vector is also the best parameter vector found so
    /// far.
    fn is_best(&self) -> bool {
        self.last_best_iter == self.iter
    }

    /// Return whether the algorithm has terminated or not
    fn terminated(&self) -> bool {
        self.termination_reason.terminated()
    }
}

/// Maintains the state from iteration to iteration of a solver
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct LinearProgramState<O: LinearProgram> {
    /// Current parameter vector
    pub param: Option<O::Param>,
    /// Previous parameter vector
    pub prev_param: Option<O::Param>,
    /// Current best parameter vector
    pub best_param: Option<O::Param>,
    /// Previous best parameter vector
    pub prev_best_param: Option<O::Param>,
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
    /// Current iteration
    pub iter: u64,
    /// Iteration number of last best cost
    pub last_best_iter: u64,
    /// Maximum number of iterations
    pub max_iters: u64,
    /// Time required so far
    pub time: Option<instant::Duration>,
    /// Reason of termination
    pub termination_reason: TerminationReason,
}

impl<O: LinearProgram> State for LinearProgramState<O> {
    type Param = O::Param;
    type Output = ();
    type Hessian = ();
    type Jacobian = ();
    type Float = O::Float;
    type Operator = O;

    /// Create new IterState from `param`
    fn new(param: Self::Param) -> Self {
        LinearProgramState {
            param: Some(param.clone()),
            prev_param: None,
            best_param: Some(param),
            prev_best_param: None,
            cost: Self::Float::infinity(),
            prev_cost: Self::Float::infinity(),
            best_cost: Self::Float::infinity(),
            prev_best_cost: Self::Float::infinity(),
            target_cost: Self::Float::neg_infinity(),
            iter: 0,
            last_best_iter: 0,
            max_iters: std::u64::MAX,
            time: Some(instant::Duration::new(0, 0)),
            termination_reason: TerminationReason::NotTerminated,
        }
    }

    fn update(&mut self, data: &ArgminIterData<LinearProgramState<O>>) {
        if let Some(cur_param) = data.get_param() {
            self.param(cur_param);
        }
        if let Some(cur_cost) = data.get_cost() {
            self.cost(cur_cost);
        }
        // check if parameters are the best so far
        // Comparison is done using `<` to avoid new solutions with the same cost function value as
        // the current best to be accepted. However, some solvers to not compute the cost function
        // value (such as the Newton method). Those will always have `Inf` cost. Therefore if both
        // the new value and the previous best value are `Inf`, the solution is also accepted. Care
        // is taken that both `Inf` also have the same sign.
        if self.get_cost() < self.get_best_cost()
            || (self.get_cost().is_infinite()
                && self.get_best_cost().is_infinite()
                && self.get_cost().is_sign_positive() == self.get_best_cost().is_sign_positive())
        {
            self.current_param_is_new_best();
        }

        if let Some(termination_reason) = data.get_termination_reason() {
            self.termination_reason(termination_reason);
        }
    }

    /// Current paramter vector is new best
    fn current_param_is_new_best(&mut self) {
        let param = self.get_param().unwrap();
        let cost = self.get_cost();
        self.best_param(param).best_cost(cost);
        self.new_best();
    }

    /// Set parameter vector. This shifts the stored parameter vector to the previous parameter
    /// vector.
    fn param(&mut self, param: Self::Param) -> &mut Self {
        std::mem::swap(&mut self.prev_param, &mut self.param);
        self.param = Some(param);
        self
    }

    /// Set best paramater vector. This shifts the stored best parameter vector to the previous
    /// best parameter vector.
    fn best_param(&mut self, param: Self::Param) -> &mut Self {
        std::mem::swap(&mut self.prev_best_param, &mut self.best_param);
        self.best_param = Some(param);
        self
    }

    /// Set the current cost function value. This shifts the stored cost function value to the
    /// previous cost function value.
    fn cost(&mut self, cost: Self::Float) -> &mut Self {
        std::mem::swap(&mut self.prev_cost, &mut self.cost);
        self.cost = cost;
        self
    }

    /// Set the current best cost function value. This shifts the stored best cost function value to
    /// the previous cost function value.
    fn best_cost(&mut self, cost: Self::Float) -> &mut Self {
        std::mem::swap(&mut self.prev_best_cost, &mut self.best_cost);
        self.best_cost = cost;
        self
    }

    setter!(target_cost, Self::Float, "Set target cost value");
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
    getter_option!(param, Self::Param, "Returns current parameter vector");
    getter_option!(prev_param, Self::Param, "Returns previous parameter vector");
    getter_option!(best_param, Self::Param, "Returns best parameter vector");
    getter_option!(
        prev_best_param,
        Self::Param,
        "Returns previous best parameter vector"
    );
    getter!(cost, Self::Float, "Returns current cost function value");
    getter!(
        prev_cost,
        Self::Float,
        "Returns previous cost function value"
    );
    getter!(
        best_cost,
        Self::Float,
        "Returns current best cost function value"
    );
    getter!(
        prev_best_cost,
        Self::Float,
        "Returns previous best cost function value"
    );
    getter!(target_cost, Self::Float, "Returns target cost");
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
    getter!(iter, u64, "Returns current number of iterations");
    getter!(max_iters, u64, "Returns maximum number of iterations");

    /// Increment the number of iterations by one
    fn increment_iter(&mut self) {
        self.iter += 1;
    }

    /// Increment all function evaluation counts by the evaluation counts of another operator
    /// wrapped in `OpWrapper`.
    fn increment_func_counts(&mut self, _op: &OpWrapper<Self::Operator>) {}

    /// Set all function evaluation counts to the evaluation counts of another operator
    /// wrapped in `OpWrapper`.
    fn set_func_counts(&mut self, _op: &OpWrapper<Self::Operator>) {}

    /// Indicate that a new best parameter vector was found
    fn new_best(&mut self) {
        self.last_best_iter = self.iter;
    }

    /// Returns whether the current parameter vector is also the best parameter vector found so
    /// far.
    fn is_best(&self) -> bool {
        self.last_best_iter == self.iter
    }

    /// Return whether the algorithm has terminated or not
    fn terminated(&self) -> bool {
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

        assert_eq!(state.get_param().unwrap(), param);
        assert_eq!(state.get_prev_param(), None);
        assert_eq!(state.get_best_param().unwrap(), param);
        assert_eq!(state.get_prev_best_param(), None);

        assert_eq!(*state.get_param_ref().unwrap(), param);
        assert!(state.get_prev_param_ref().is_none());
        assert_eq!(*state.get_best_param_ref().unwrap(), param);
        assert!(state.get_prev_best_param_ref().is_none());

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

        assert!(state.get_grad().is_none());
        assert!(state.get_prev_grad().is_none());
        assert!(state.get_hessian().is_none());
        assert!(state.get_prev_hessian().is_none());
        assert!(state.get_inv_hessian().is_none());
        assert!(state.get_prev_inv_hessian().is_none());
        assert!(state.get_jacobian().is_none());
        assert!(state.get_prev_jacobian().is_none());
        assert!(state.get_grad_ref().is_none());
        assert!(state.get_prev_grad_ref().is_none());
        assert!(state.get_hessian_ref().is_none());
        assert!(state.get_prev_hessian_ref().is_none());
        assert!(state.get_inv_hessian_ref().is_none());
        assert!(state.get_prev_inv_hessian_ref().is_none());
        assert!(state.get_jacobian_ref().is_none());
        assert!(state.get_prev_jacobian_ref().is_none());
        assert_eq!(state.get_iter(), 0);

        assert!(state.is_best());

        assert_eq!(state.get_max_iters(), std::u64::MAX);
        assert_eq!(state.get_cost_func_count(), 0);
        assert_eq!(state.get_grad_func_count(), 0);
        assert_eq!(state.get_hessian_func_count(), 0);
        assert_eq!(state.get_jacobian_func_count(), 0);
        assert_eq!(state.get_modify_func_count(), 0);

        state.max_iters(42);

        assert_eq!(state.get_max_iters(), 42);

        state.cost(cost);

        assert_eq!(state.get_cost().to_ne_bytes(), cost.to_ne_bytes());
        assert!(state.get_prev_cost().is_infinite());
        assert!(state.get_prev_cost().is_sign_positive());

        state.best_cost(cost);

        assert_eq!(state.get_best_cost().to_ne_bytes(), cost.to_ne_bytes());
        assert!(state.get_prev_best_cost().is_infinite());
        assert!(state.get_prev_best_cost().is_sign_positive());

        let new_param = vec![2.0, 1.0];

        state.param(new_param.clone());

        assert_eq!(state.get_param().unwrap(), new_param);
        assert_eq!(state.get_prev_param().unwrap(), param);
        assert_eq!(*state.get_param_ref().unwrap(), new_param);
        assert_eq!(*state.get_prev_param_ref().unwrap(), param);

        state.best_param(new_param.clone());

        assert_eq!(state.get_best_param().unwrap(), new_param);
        assert_eq!(state.get_prev_best_param().unwrap(), param);
        assert_eq!(*state.get_best_param_ref().unwrap(), new_param);
        assert_eq!(*state.get_prev_best_param_ref().unwrap(), param);

        let new_cost = 21.0;

        state.cost(new_cost);

        assert_eq!(state.get_cost().to_ne_bytes(), new_cost.to_ne_bytes());
        assert_eq!(state.get_prev_cost().to_ne_bytes(), cost.to_ne_bytes());

        state.best_cost(new_cost);

        assert_eq!(state.get_best_cost().to_ne_bytes(), new_cost.to_ne_bytes());
        assert_eq!(state.get_prev_best_cost().to_ne_bytes(), cost.to_ne_bytes());

        state.increment_iter();

        assert_eq!(state.get_iter(), 1);

        assert!(!state.is_best());

        state.new_best();

        assert!(state.is_best());

        let grad = vec![1.0, 2.0];

        state.grad(grad.clone());
        assert_eq!(state.get_grad().unwrap(), grad);
        assert!(state.get_prev_grad().is_none());
        assert_eq!(*state.get_grad_ref().unwrap(), grad);
        assert!(state.get_prev_grad_ref().is_none());

        let new_grad = vec![2.0, 1.0];

        state.grad(new_grad.clone());

        assert_eq!(state.get_grad().unwrap(), new_grad);
        assert_eq!(state.get_prev_grad().unwrap(), grad);
        assert_eq!(*state.get_grad_ref().unwrap(), new_grad);
        assert_eq!(*state.get_prev_grad_ref().unwrap(), grad);

        let hessian = vec![vec![1.0, 2.0], vec![2.0, 1.0]];

        state.hessian(hessian.clone());
        assert_eq!(state.get_hessian().unwrap(), hessian);
        assert!(state.get_prev_hessian().is_none());
        assert_eq!(*state.get_hessian_ref().unwrap(), hessian);
        assert!(state.get_prev_hessian_ref().is_none());

        let new_hessian = vec![vec![2.0, 1.0], vec![1.0, 2.0]];

        state.hessian(new_hessian.clone());

        assert_eq!(state.get_hessian().unwrap(), new_hessian);
        assert_eq!(state.get_prev_hessian().unwrap(), hessian);
        assert_eq!(*state.get_hessian_ref().unwrap(), new_hessian);
        assert_eq!(*state.get_prev_hessian_ref().unwrap(), hessian);

        let inv_hessian = vec![vec![2.0, 1.0], vec![1.0, 2.0]];

        state.inv_hessian(inv_hessian.clone());
        assert_eq!(state.get_inv_hessian().unwrap(), inv_hessian);
        assert!(state.get_prev_inv_hessian().is_none());
        assert_eq!(*state.get_inv_hessian_ref().unwrap(), inv_hessian);
        assert!(state.get_prev_inv_hessian_ref().is_none());

        let new_inv_hessian = vec![vec![3.0, 4.0], vec![4.0, 3.0]];

        state.inv_hessian(new_inv_hessian.clone());

        assert_eq!(state.get_inv_hessian().unwrap(), new_inv_hessian);
        assert_eq!(state.get_prev_inv_hessian().unwrap(), inv_hessian);
        assert_eq!(*state.get_inv_hessian_ref().unwrap(), new_inv_hessian);
        assert_eq!(*state.get_prev_inv_hessian_ref().unwrap(), inv_hessian);

        let jacobian = vec![1.0, 2.0];

        state.jacobian(jacobian.clone());
        assert_eq!(state.get_jacobian().unwrap(), jacobian);
        assert!(state.get_prev_jacobian().is_none());
        assert_eq!(*state.get_jacobian_ref().unwrap(), jacobian);
        assert!(state.get_prev_jacobian_ref().is_none());

        let new_jacobian = vec![2.0, 1.0];

        state.jacobian(new_jacobian.clone());

        assert_eq!(state.get_jacobian().unwrap(), new_jacobian);
        assert_eq!(state.get_prev_jacobian().unwrap(), jacobian);
        assert_eq!(*state.get_jacobian_ref().unwrap(), new_jacobian);
        assert_eq!(*state.get_prev_jacobian_ref().unwrap(), jacobian);

        state.increment_iter();

        assert_eq!(state.get_iter(), 2);
        assert_eq!(state.get_last_best_iter(), 1);
        assert!(!state.is_best());

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

        assert!(!state.is_best());

        assert_eq!(state.get_cost().to_ne_bytes(), new_cost.to_ne_bytes());
        assert_eq!(state.get_prev_cost().to_ne_bytes(), cost.to_ne_bytes());
        assert_eq!(state.get_prev_cost().to_ne_bytes(), cost.to_ne_bytes());

        assert_eq!(state.get_param().unwrap(), new_param);
        assert_eq!(state.get_prev_param().unwrap(), param);
        assert_eq!(*state.get_param_ref().unwrap(), new_param);
        assert_eq!(*state.get_prev_param_ref().unwrap(), param);

        assert_eq!(state.get_best_cost().to_ne_bytes(), new_cost.to_ne_bytes());
        assert_eq!(state.get_prev_best_cost().to_ne_bytes(), cost.to_ne_bytes());

        assert_eq!(state.get_best_param().unwrap(), new_param);
        assert_eq!(state.get_prev_best_param().unwrap(), param);
        assert_eq!(*state.get_best_param_ref().unwrap(), new_param);
        assert_eq!(*state.get_prev_best_param_ref().unwrap(), param);

        assert_eq!(state.get_best_cost().to_ne_bytes(), new_cost.to_ne_bytes());
        assert_eq!(state.get_prev_best_cost().to_ne_bytes(), cost.to_ne_bytes());

        assert_eq!(state.get_grad().unwrap(), new_grad);
        assert_eq!(state.get_prev_grad().unwrap(), grad);
        assert_eq!(state.get_hessian().unwrap(), new_hessian);
        assert_eq!(state.get_prev_hessian().unwrap(), hessian);
        assert_eq!(state.get_inv_hessian().unwrap(), new_inv_hessian);
        assert_eq!(state.get_prev_inv_hessian().unwrap(), inv_hessian);
        assert_eq!(state.get_jacobian().unwrap(), new_jacobian);
        assert_eq!(state.get_prev_jacobian().unwrap(), jacobian);
        assert_eq!(*state.get_grad_ref().unwrap(), new_grad);
        assert_eq!(*state.get_prev_grad_ref().unwrap(), grad);
        assert_eq!(*state.get_hessian_ref().unwrap(), new_hessian);
        assert_eq!(*state.get_prev_hessian_ref().unwrap(), hessian);
        assert_eq!(*state.get_inv_hessian_ref().unwrap(), new_inv_hessian);
        assert_eq!(*state.get_prev_inv_hessian_ref().unwrap(), inv_hessian);
        assert_eq!(*state.get_jacobian_ref().unwrap(), new_jacobian);
        assert_eq!(*state.get_prev_jacobian_ref().unwrap(), jacobian);
        assert_eq!(state.take_grad().unwrap(), new_grad);
        assert_eq!(state.take_prev_grad().unwrap(), grad);
        assert_eq!(state.take_hessian().unwrap(), new_hessian);
        assert_eq!(state.take_prev_hessian().unwrap(), hessian);
        assert_eq!(state.take_inv_hessian().unwrap(), new_inv_hessian);
        assert_eq!(state.take_prev_inv_hessian().unwrap(), inv_hessian);
        assert_eq!(state.take_jacobian().unwrap(), new_jacobian);
        assert_eq!(state.take_prev_jacobian().unwrap(), jacobian);
        assert!(state.get_grad().is_none());
        assert!(state.get_prev_grad().is_none());
        assert!(state.get_hessian().is_none());
        assert!(state.get_prev_hessian().is_none());
        assert!(state.get_inv_hessian().is_none());
        assert!(state.get_prev_inv_hessian().is_none());
        assert!(state.get_jacobian().is_none());
        assert!(state.get_prev_jacobian().is_none());
        assert_eq!(state.get_cost_func_count(), 42);
        assert_eq!(state.get_grad_func_count(), 43);
        assert_eq!(state.get_hessian_func_count(), 44);
        assert_eq!(state.get_jacobian_func_count(), 46);
        assert_eq!(state.get_modify_func_count(), 45);

        let old_best = vec![1.0, 2.0];
        let old_cost = 10.0;
        state.best_param(old_best.clone());
        state.best_cost(old_cost);
        let new_param = vec![3.0, 4.0];
        let new_cost = 5.0;
        state.param(new_param.clone());
        state.cost(new_cost);

        state.current_param_is_new_best();

        assert_eq!(state.get_best_param().unwrap(), new_param);
        assert_eq!(state.get_prev_best_param().unwrap(), old_best);
        assert_eq!(state.get_best_cost().to_ne_bytes(), new_cost.to_ne_bytes());
        assert_eq!(
            state.get_prev_best_cost().to_ne_bytes(),
            old_cost.to_ne_bytes()
        );
    }
}
