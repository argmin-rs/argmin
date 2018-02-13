// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! `ArgminResult`
//!
//! TODO Documentation

use parameter::ArgminParameter;
use ArgminCostValue;
use termination::TerminationReason;

/// Return struct for all solvers.
#[derive(Debug, Clone)]
pub struct ArgminResult<T: ArgminParameter, U: ArgminCostValue> {
    /// Final parameter vector
    pub param: T,
    /// Final cost value
    pub cost: U,
    /// Number of iterations
    pub iters: u64,
    /// Indicated whether it terminated or not
    pub terminated: bool,
    /// Reason of termination
    pub termination_reason: TerminationReason,
}

impl<T: ArgminParameter, U: ArgminCostValue> ArgminResult<T, U> {
    /// Constructor
    ///
    /// `param`: Final (best) parameter vector
    /// `cost`: Final (best) cost function value
    /// `iters`: Number of iterations
    pub fn new(param: T, cost: U, iters: u64) -> Self {
        ArgminResult {
            param: param,
            cost: cost,
            iters: iters,
            terminated: false,
            termination_reason: TerminationReason::NotTerminated,
        }
    }

    /// Set the termination reason
    ///
    /// In case of `NotTerminated`, the field `terminated` is set to `false` and `true` otherwise.
    ///
    /// `termination_reason`: Termination reason of type `TerminationReason`
    pub fn set_termination_reason(&mut self, termination_reason: TerminationReason) -> &mut Self {
        self.termination_reason = termination_reason;
        self.terminated = match self.termination_reason {
            TerminationReason::NotTerminated => false,
            _ => true,
        };
        self
    }
}

unsafe impl<T: ArgminParameter, U: ArgminCostValue> Send for ArgminResult<T, U> {}
unsafe impl<T: ArgminParameter, U: ArgminCostValue> Sync for ArgminResult<T, U> {}
