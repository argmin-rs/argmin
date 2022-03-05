// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! TODO: Documentation

use crate::core::{ArgminFloat, OpWrapper, State, TerminationReason};
use crate::{getter, getter_option_ref, setter};
use instant;
use paste::item;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Maintains the state from iteration to iteration of a solver
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct LinearProgramState<P, F> {
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

impl<P, F> LinearProgramState<P, F> {
    /// Set parameter vector. This shifts the stored parameter vector to the previous parameter
    /// vector.
    #[must_use]
    pub fn param(mut self, param: P) -> Self {
        std::mem::swap(&mut self.prev_param, &mut self.param);
        self.param = Some(param);
        self
    }

    /// Set target cost
    #[must_use]
    pub fn target_cost(mut self, target_cost: F) -> Self {
        self.target_cost = target_cost;
        self
    }

    /// Set best paramater vector. This shifts the stored best parameter vector to the previous
    /// best parameter vector.
    fn best_param(&mut self, param: P) -> &mut Self {
        std::mem::swap(&mut self.prev_best_param, &mut self.best_param);
        self.best_param = Some(param);
        self
    }

    /// Set the current best cost function value. This shifts the stored best cost function value to
    /// the previous cost function value.
    fn best_cost(&mut self, cost: F) -> &mut Self {
        std::mem::swap(&mut self.prev_best_cost, &mut self.best_cost);
        self.best_cost = cost;
        self
    }

    /// Set maximum number of iterations
    #[must_use]
    pub fn max_iters(mut self, iters: u64) -> Self {
        self.max_iters = iters;
        self
    }

    /// Indicate that a new best parameter vector was found
    fn new_best(&mut self) {
        self.last_best_iter = self.iter;
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

impl<P, F> State for LinearProgramState<P, F>
where
    P: Clone,
    F: ArgminFloat,
{
    /// Type of parameter vector
    type Param = P;
    /// Floating point precision
    type Float = F;

    /// Create new LinearProgramState
    fn new() -> Self {
        LinearProgramState {
            param: None,
            prev_param: None,
            best_param: None,
            prev_best_param: None,
            cost: Self::Float::infinity(),
            prev_cost: Self::Float::infinity(),
            best_cost: Self::Float::infinity(),
            prev_best_cost: Self::Float::infinity(),
            target_cost: Self::Float::neg_infinity(),
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

    getter_option_ref!(best_param, P, "Returns reference to best parameter vector");

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
