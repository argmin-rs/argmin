// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::{
    ArgminFloat,
    // LinearProgram
    // DeserializeOwnedAlias,
    OpWrapper,
    // SerializeAlias,
    TerminationReason,
};
use crate::{getter, pub_getter, pub_getter_option_ref, pub_take, setter};
use instant;
use paste::item;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;

/// Types implemeting this trait can be used to keep track of a solver's state
pub trait State: Debug + Sized {
    /// Floating Point Precision
    type Float: ArgminFloat;

    /// Constructor
    fn new() -> Self;

    /// TODO
    fn update(&mut self);

    /// Returns maximum number of iterations
    fn get_max_iters(&self) -> u64;

    /// Increment the number of iterations by one
    fn increment_iter(&mut self);

    /// Returns current number of iterations
    fn get_iter(&self) -> u64;

    /// Returns current cost function value
    fn get_cost(&self) -> Self::Float;

    /// Returns best cost function value
    fn get_best_cost(&self) -> Self::Float;

    /// Returns target cost
    fn get_target_cost(&self) -> Self::Float;

    /// Set all function evaluation counts to the evaluation counts of another operator
    /// wrapped in `OpWrapper`.
    fn set_func_counts<O>(&mut self, op: &OpWrapper<O>);

    /// Return whether the algorithm has terminated or not
    fn terminated(&self) -> bool;

    /// Set termination reason
    #[must_use]
    fn termination_reason(self, termination_reason: TerminationReason) -> Self;

    /// Returns termination reason
    fn get_termination_reason(&self) -> TerminationReason;

    /// Set time required so far
    fn time(&mut self, time: Option<instant::Duration>) -> &mut Self;

    /// Get time required so far
    fn get_time(&self) -> Option<instant::Duration>;

    /// Returns iteration number where the last best parameter vector was found
    fn get_last_best_iter(&self) -> u64;

    /// Returns whether the current parameter vector is also the best parameter vector found so
    /// far.
    fn is_best(&self) -> bool;

    /// Returns currecnt cost function evaluation count
    fn get_func_counts(&self) -> &HashMap<String, u64>;
}
impl<P, G, J, H, F> std::fmt::Debug for IterState<P, G, J, H, F>
where
    Self: State<Float = F>,
    F: ArgminFloat,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        // TODO!
        writeln!(f, "{:?}", self.best_cost)?;
        Ok(())
    }
}

/// Maintains the state from iteration to iteration of a solver
#[derive(Clone, Default)]
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
    /// All members for population-based algorithms as (param, cost) tuples
    pub population: Option<Vec<(P, F)>>,
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
    #[must_use]
    pub fn param(mut self, param: P) -> Self {
        std::mem::swap(&mut self.prev_param, &mut self.param);
        self.param = Some(param);
        self
    }

    /// Set best paramater vector. This shifts the stored best parameter vector to the previous
    /// best parameter vector.
    fn best_param(&mut self, param: P) -> &mut Self {
        std::mem::swap(&mut self.prev_best_param, &mut self.best_param);
        self.best_param = Some(param);
        self
    }

    /// Set target cost
    #[must_use]
    pub fn target_cost(mut self, target_cost: F) -> Self {
        self.target_cost = target_cost;
        self
    }

    /// Set the current best cost function value. This shifts the stored best cost function value to
    /// the previous cost function value.
    fn best_cost(&mut self, cost: F) -> &mut Self {
        std::mem::swap(&mut self.prev_best_cost, &mut self.best_cost);
        self.best_cost = cost;
        self
    }

    /// Set population
    #[must_use]
    pub fn population(mut self, population: Vec<(P, F)>) -> Self {
        self.population = Some(population);
        self
    }

    /// Set maximum number of iterations
    #[must_use]
    pub fn max_iters(mut self, iters: u64) -> Self {
        self.max_iters = iters;
        self
    }

    pub_getter!(cost, F, "Returns current cost function value");
    pub_getter!(best_cost, F, "Returns current best cost function value");
    pub_getter!(target_cost, F, "Returns current best cost function value");
    pub_getter_option_ref!(param, P, "Returns reference to current parameter vector");
    pub_take!(
        param,
        P,
        "Moves the current parameter vector out and replaces it internally with `None`"
    );
    pub_getter_option_ref!(
        prev_param,
        P,
        "Returns reference to previous parameter vector"
    );
    pub_take!(
        prev_param,
        P,
        "Moves the previous parameter vector out and replaces it internally with `None`"
    );
    pub_getter_option_ref!(best_param, P, "Returns reference to best parameter vector");
    pub_getter_option_ref!(
        prev_best_param,
        P,
        "Returns reference to previous best parameter vector"
    );
    pub_take!(
        best_param,
        P,
        "Moves the best parameter vector out and replaces it internally with `None`"
    );
    pub_take!(
        prev_best_param,
        P,
        "Moves the previous best parameter vector out and replaces it internally with `None`"
    );
    pub_getter!(prev_cost, F, "Returns previous cost function value");
    pub_getter!(
        prev_best_cost,
        F,
        "Returns previous best cost function value"
    );
    pub_getter_option_ref!(grad, G, "Returns reference to the gradient");
    pub_getter_option_ref!(prev_grad, G, "Returns reference to the previous gradient");
    pub_getter_option_ref!(hessian, H, "Returns reference to the current Hessian");
    pub_getter_option_ref!(prev_hessian, H, "Returns reference to the previous Hessian");
    pub_getter_option_ref!(jacobian, J, "Returns reference to the current Jacobian");
    pub_getter_option_ref!(
        prev_jacobian,
        J,
        "Returns reference to the previous Jacobian"
    );
    pub_getter_option_ref!(
        inv_hessian,
        H,
        "Returns reference to the current inverse Hessian"
    );
    pub_getter_option_ref!(
        prev_inv_hessian,
        H,
        "Returns reference to the previous inverse Hessian"
    );
    pub_take!(
        grad,
        G,
        "Moves the gradient out and replaces it internally with `None`"
    );
    pub_take!(
        prev_grad,
        G,
        "Moves the previous gradient out and replaces it internally with `None`"
    );
    pub_take!(
        hessian,
        H,
        "Moves the Hessian out and replaces it internally with `None`"
    );
    pub_take!(
        prev_hessian,
        H,
        "Moves the previous Hessian out and replaces it internally with `None`"
    );
    pub_take!(
        jacobian,
        J,
        "Moves the Jacobian out and replaces it internally with `None`"
    );
    pub_take!(
        prev_jacobian,
        J,
        "Moves the previous Jacobian out and replaces it internally with `None`"
    );
    pub_take!(
        inv_hessian,
        H,
        "Moves the inverse Hessian out and replaces it internally with `None`"
    );
    pub_take!(
        prev_inv_hessian,
        H,
        "Moves the previous inverse Hessian out and replaces it internally with `None`"
    );

    /// Returns population
    pub fn get_population(&self) -> Option<&Vec<(P, F)>> {
        self.population.as_ref()
    }

    /// Indicate that a new best parameter vector was found
    fn new_best(&mut self) {
        self.last_best_iter = self.iter;
    }

    /// Set gradient. This shifts the stored gradient to the previous gradient.
    #[must_use]
    pub fn grad(mut self, grad: G) -> Self {
        std::mem::swap(&mut self.prev_grad, &mut self.grad);
        self.grad = Some(grad);
        self
    }

    /// Set Hessian. This shifts the stored Hessian to the previous Hessian.
    #[must_use]
    pub fn hessian(mut self, hessian: H) -> Self {
        std::mem::swap(&mut self.prev_hessian, &mut self.hessian);
        self.hessian = Some(hessian);
        self
    }

    /// Set inverse Hessian. This shifts the stored inverse Hessian to the previous inverse Hessian.
    #[must_use]
    pub fn inv_hessian(mut self, inv_hessian: H) -> Self {
        std::mem::swap(&mut self.prev_inv_hessian, &mut self.inv_hessian);
        self.inv_hessian = Some(inv_hessian);
        self
    }

    /// Set Jacobian. This shifts the stored Jacobian to the previous Jacobian.
    #[must_use]
    pub fn jacobian(mut self, jacobian: J) -> Self {
        std::mem::swap(&mut self.prev_jacobian, &mut self.jacobian);
        self.jacobian = Some(jacobian);
        self
    }

    /// Set the current cost function value. This shifts the stored cost function value to the
    /// previous cost function value.
    #[must_use]
    pub fn cost(mut self, cost: F) -> Self {
        std::mem::swap(&mut self.prev_cost, &mut self.cost);
        self.cost = cost;
        self
    }
}

impl<P, G, J, H, F> State for IterState<P, G, J, H, F>
where
    P: Clone,
    F: ArgminFloat,
{
    /// Floating point precision
    type Float = F;

    /// Create new IterState from `param`
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
            population: None,
            iter: 0,
            last_best_iter: 0,
            max_iters: std::u64::MAX,
            counts: HashMap::new(),
            time: Some(instant::Duration::new(0, 0)),
            termination_reason: TerminationReason::NotTerminated,
        }
    }

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
            let param = (*self.param.as_ref().unwrap()).clone();
            let cost = self.cost;
            self.best_param(param).best_cost(cost);
            self.new_best();
        }
    }

    #[must_use]
    fn termination_reason(mut self, reason: TerminationReason) -> Self {
        self.termination_reason = reason;
        self
    }

    setter!(time, Option<instant::Duration>, "Set time required so far");
    getter!(cost, Self::Float, "Returns current cost function value");
    getter!(
        best_cost,
        Self::Float,
        "Returns current best cost function value"
    );
    getter!(
        target_cost,
        Self::Float,
        "Returns current best cost function value"
    );
    getter!(iter, u64, "Returns current number of iterations");
    getter!(max_iters, u64, "Returns maximum number of iterations");
    getter!(
        termination_reason,
        TerminationReason,
        "Get termination_reason"
    );
    getter!(time, Option<instant::Duration>, "Get time required so far");
    getter!(
        last_best_iter,
        u64,
        "Returns iteration number where the last best parameter vector was found"
    );

    /// Increment the number of iterations by one
    fn increment_iter(&mut self) {
        self.iter += 1;
    }

    /// Set all function evaluation counts to the evaluation counts of another operator
    /// wrapped in `OpWrapper`.
    fn set_func_counts<O>(&mut self, op: &OpWrapper<O>) {
        for (k, &v) in op.counts.iter() {
            let count = self.counts.entry(k.to_string()).or_insert(0);
            *count = v
        }
    }

    fn get_func_counts(&self) -> &HashMap<String, u64> {
        &self.counts
    }

    /// Return whether the algorithm has terminated or not
    fn terminated(&self) -> bool {
        self.termination_reason.terminated()
    }

    /// Returns whether the current parameter vector is also the best parameter vector found so
    /// far.
    fn is_best(&self) -> bool {
        self.last_best_iter == self.iter
    }
}
//
// /// Maintains the state from iteration to iteration of a solver
// #[derive(Clone, Debug)]
// #[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
// pub struct LinearProgramState<O: LinearProgram> {
//     /// Current parameter vector
//     pub param: Option<P>,
//     /// Previous parameter vector
//     pub prev_param: Option<P>,
//     /// Current best parameter vector
//     pub best_param: Option<P>,
//     /// Previous best parameter vector
//     pub prev_best_param: Option<P>,
//     /// Current cost function value
//     pub cost: F,
//     /// Previous cost function value
//     pub prev_cost: F,
//     /// Current best cost function value
//     pub best_cost: F,
//     /// Previous best cost function value
//     pub prev_best_cost: F,
//     /// Target cost function value
//     pub target_cost: F,
//     /// Current iteration
//     pub iter: u64,
//     /// Iteration number of last best cost
//     pub last_best_iter: u64,
//     /// Maximum number of iterations
//     pub max_iters: u64,
//     /// Time required so far
//     pub time: Option<instant::Duration>,
//     /// Reason of termination
//     pub termination_reason: TerminationReason,
// }
//
// impl<O: LinearProgram> LinearProgramState<O> {
//     /// Set parameter vector. This shifts the stored parameter vector to the previous parameter
//     /// vector.
//     #[must_use]
//     pub fn param(mut self, param: P) -> Self {
//         std::mem::swap(&mut self.prev_param, &mut self.param);
//         self.param = Some(param);
//         self
//     }
//
//     /// Set target cost
//     #[must_use]
//     pub fn target_cost(mut self, target_cost: F) -> Self {
//         self.target_cost = target_cost;
//         self
//     }
//
//     /// Set best paramater vector. This shifts the stored best parameter vector to the previous
//     /// best parameter vector.
//     fn best_param(&mut self, param: P) -> &mut Self {
//         std::mem::swap(&mut self.prev_best_param, &mut self.best_param);
//         self.best_param = Some(param);
//         self
//     }
//
//     /// Set the current best cost function value. This shifts the stored best cost function value to
//     /// the previous cost function value.
//     fn best_cost(&mut self, cost: F) -> &mut Self {
//         std::mem::swap(&mut self.prev_best_cost, &mut self.best_cost);
//         self.best_cost = cost;
//         self
//     }
//
//     /// Set maximum number of iterations
//     #[must_use]
//     pub fn max_iters(mut self, iters: u64) -> Self {
//         self.max_iters = iters;
//         self
//     }
//
//     // setter!(
//     //     last_best_iter,
//     //     u64,
//     //     "Set iteration number where the previous best parameter vector was found"
//     // );
//     // getter_option!(prev_param, P, "Returns previous parameter vector");
//     // getter_option!(
//     //     prev_best_param,
//     //     P,
//     //     "Returns previous best parameter vector"
//     // );
//     // getter!(prev_cost, F, "Returns previous cost function value");
//     // getter!(
//     //     prev_best_cost,
//     //     F,
//     //     "Returns previous best cost function value"
//     // );
//     /// Indicate that a new best parameter vector was found
//     fn new_best(&mut self) {
//         self.last_best_iter = self.iter;
//     }
//
//     /// Set the current cost function value. This shifts the stored cost function value to the
//     /// previous cost function value.
//     #[must_use]
//     pub fn cost(mut self, cost: F) -> Self {
//         std::mem::swap(&mut self.prev_cost, &mut self.cost);
//         self.cost = cost;
//         self
//     }
// }
//
// impl<O: LinearProgram> State for LinearProgramState<O> {
//     // type Param = P;
//     // type Output = ();
//     // type Hessian = ();
//     // type Jacobian = ();
//     // type Float = F;
//     type Operator = O;
//
//     /// Create new IterState from `param`
//     fn new() -> Self {
//         LinearProgramState {
//             param: None,
//             prev_param: None,
//             best_param: None,
//             prev_best_param: None,
//             cost: Self::Float::infinity(),
//             prev_cost: Self::Float::infinity(),
//             best_cost: Self::Float::infinity(),
//             prev_best_cost: Self::Float::infinity(),
//             target_cost: Self::Float::neg_infinity(),
//             iter: 0,
//             last_best_iter: 0,
//             max_iters: std::u64::MAX,
//             time: Some(instant::Duration::new(0, 0)),
//             termination_reason: TerminationReason::NotTerminated,
//         }
//     }
//
//     fn update(&mut self, data: &ArgminIterData<O>) {
//         if let Some(cur_param) = data.get_param() {
//             std::mem::swap(&mut self.prev_param, &mut self.param);
//             self.param = Some(cur_param);
//         }
//         if let Some(cur_cost) = data.get_cost() {
//             std::mem::swap(&mut self.prev_cost, &mut self.cost);
//             self.cost = cur_cost;
//         }
//         // check if parameters are the best so far
//         // Comparison is done using `<` to avoid new solutions with the same cost function value as
//         // the current best to be accepted. However, some solvers to not compute the cost function
//         // value (such as the Newton method). Those will always have `Inf` cost. Therefore if both
//         // the new value and the previous best value are `Inf`, the solution is also accepted. Care
//         // is taken that both `Inf` also have the same sign.
//         if self.get_cost() < self.get_best_cost()
//             || (self.get_cost().is_infinite()
//                 && self.get_best_cost().is_infinite()
//                 && self.get_cost().is_sign_positive() == self.get_best_cost().is_sign_positive())
//         {
//             let param = self.get_param().unwrap();
//             let cost = self.get_cost();
//             self.best_param(param).best_cost(cost);
//             self.new_best();
//         }
//
//         if let Some(termination_reason) = data.get_termination_reason() {
//             self.termination_reason(termination_reason);
//         }
//     }
//
//     setter!(
//         termination_reason,
//         TerminationReason,
//         "Set termination_reason"
//     );
//     setter!(time, Option<instant::Duration>, "Set time required so far");
//     // getter_option!(param, Self::Param, "Returns current parameter vector");
//     // getter_option!(best_param, Self::Param, "Returns best parameter vector");
//     // getter!(cost, Self::Float, "Returns current cost function value");
//     // getter!(
//     //     best_cost,
//     //     Self::Float,
//     //     "Returns current best cost function value"
//     // );
//     // getter!(target_cost, Self::Float, "Returns target cost");
//     getter!(
//         last_best_iter,
//         u64,
//         "Returns iteration number where the last best parameter vector was found"
//     );
//     getter!(
//         termination_reason,
//         TerminationReason,
//         "Get termination_reason"
//     );
//     getter!(time, Option<instant::Duration>, "Get time required so far");
//     getter!(iter, u64, "Returns current number of iterations");
//     getter!(max_iters, u64, "Returns maximum number of iterations");
//
//     /// Increment the number of iterations by one
//     fn increment_iter(&mut self) {
//         self.iter += 1;
//     }
//     //
//     // /// Set all function evaluation counts to the evaluation counts of another operator
//     // /// wrapped in `OpWrapper`.
//     // fn set_func_counts(&mut self, _op: &OpWrapper<Self::Operator>) {}
//     //
//     /// Returns whether the current parameter vector is also the best parameter vector found so
//     /// far.
//     fn is_best(&self) -> bool {
//         self.last_best_iter == self.iter
//     }
//
//     /// Return whether the algorithm has terminated or not
//     fn terminated(&self) -> bool {
//         self.termination_reason.terminated()
//     }
//
//     /// Returns currecnt cost function evaluation count
//     fn get_cost_func_count(&self) -> u64 {
//         0
//     }
//
//     /// Returns current gradient function evaluation count
//     fn get_grad_func_count(&self) -> u64 {
//         0
//     }
//
//     /// Returns current Hessian function evaluation count
//     fn get_hessian_func_count(&self) -> u64 {
//         0
//     }
//
//     /// Returns current Jacobian function evaluation count
//     fn get_jacobian_func_count(&self) -> u64 {
//         0
//     }
//
//     /// Returns current modify function evaluation count
//     fn get_modify_func_count(&self) -> u64 {
//         0
//     }
// }
//
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

        assert!(state.get_param_ref().is_none());
        assert!(state.get_prev_param_ref().is_none());
        assert!(state.get_best_param_ref().is_none());
        assert!(state.get_prev_best_param_ref().is_none());

        state = state.param(param.clone());

        assert_eq!(*state.get_param_ref().unwrap(), param);
        assert!(state.get_prev_param_ref().is_none());
        assert!(state.get_best_param_ref().is_none());
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

        state.best_cost(cost);

        assert_eq!(state.get_best_cost().to_ne_bytes(), cost.to_ne_bytes());
        assert!(state.get_prev_best_cost().is_infinite());
        assert!(state.get_prev_best_cost().is_sign_positive());

        let new_param = vec![2.0, 1.0];

        state = state.param(new_param.clone());

        assert_eq!(*state.get_param_ref().unwrap(), new_param);
        assert_eq!(*state.get_prev_param_ref().unwrap(), param);

        state.best_param(new_param.clone());

        assert_eq!(*state.get_best_param_ref().unwrap(), new_param);
        assert!(state.get_prev_best_param_ref().is_none());

        let new_cost: f64 = 21.0;

        let mut state = state.cost(new_cost);

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

        let state = state.grad(grad.clone());
        assert_eq!(*state.get_grad_ref().unwrap(), grad);
        assert!(state.get_prev_grad_ref().is_none());

        let new_grad = vec![2.0, 1.0];

        let state = state.grad(new_grad.clone());

        assert_eq!(*state.get_grad_ref().unwrap(), new_grad);
        assert_eq!(*state.get_prev_grad_ref().unwrap(), grad);

        let hessian = vec![vec![1.0, 2.0], vec![2.0, 1.0]];

        let state = state.hessian(hessian.clone());
        assert_eq!(*state.get_hessian_ref().unwrap(), hessian);
        assert!(state.get_prev_hessian_ref().is_none());

        let new_hessian = vec![vec![2.0, 1.0], vec![1.0, 2.0]];

        let state = state.hessian(new_hessian.clone());

        assert_eq!(*state.get_hessian_ref().unwrap(), new_hessian);
        assert_eq!(*state.get_prev_hessian_ref().unwrap(), hessian);

        let inv_hessian = vec![vec![2.0, 1.0], vec![1.0, 2.0]];

        let state = state.inv_hessian(inv_hessian.clone());
        assert_eq!(*state.get_inv_hessian_ref().unwrap(), inv_hessian);
        assert!(state.get_prev_inv_hessian_ref().is_none());

        let new_inv_hessian = vec![vec![3.0, 4.0], vec![4.0, 3.0]];

        let state = state.inv_hessian(new_inv_hessian.clone());

        assert_eq!(*state.get_inv_hessian_ref().unwrap(), new_inv_hessian);
        assert_eq!(*state.get_prev_inv_hessian_ref().unwrap(), inv_hessian);

        let jacobian = vec![1.0f64, 2.0];

        let state = state.jacobian(jacobian.clone());
        assert!(state.get_prev_jacobian_ref().is_none());

        let new_jacobian = vec![2.0f64, 1.0];

        let mut state = state.jacobian(new_jacobian.clone());

        assert_eq!(*state.get_jacobian_ref().unwrap(), new_jacobian);
        assert_eq!(*state.get_prev_jacobian_ref().unwrap(), jacobian);

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

        assert_eq!(*state.get_param_ref().unwrap(), new_param);
        assert_eq!(*state.get_prev_param_ref().unwrap(), param);

        assert_eq!(state.get_best_cost().to_ne_bytes(), new_cost.to_ne_bytes());
        assert_eq!(state.get_prev_best_cost().to_ne_bytes(), cost.to_ne_bytes());

        assert_eq!(*state.get_best_param_ref().unwrap(), new_param);
        assert!(state.get_prev_best_param_ref().is_none());

        assert_eq!(state.get_best_cost().to_ne_bytes(), new_cost.to_ne_bytes());
        assert_eq!(state.get_prev_best_cost().to_ne_bytes(), cost.to_ne_bytes());

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
        let func_counts = state.get_func_counts().clone();
        assert!(!func_counts.contains_key("cost_count"));
        assert!(!func_counts.contains_key("operator_count"));
        assert!(!func_counts.contains_key("gradient_count"));
        assert!(!func_counts.contains_key("hessian_count"));
        assert!(!func_counts.contains_key("jacobian_count"));
        assert!(!func_counts.contains_key("modify_count"));

        let old_best = vec![1.0, 2.0];
        let old_cost: f64 = 10.0;
        state.best_param(old_best);
        state.best_cost(old_cost);
        let new_param = vec![3.0, 4.0];
        let new_cost: f64 = 5.0;
        state = state.param(new_param);
        let _state = state.cost(new_cost);

        // TODO: Test update
    }
}
