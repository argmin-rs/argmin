// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::{Error, Problem, SerializeAlias, State, TerminationReason, KV};

/// Solver
///
/// Every solver needs to implement this trait.
pub trait Solver<O, I: State>: SerializeAlias {
    /// Name of the solver
    const NAME: &'static str = "UNDEFINED";

    /// Computes one iteration of the algorithm.
    fn next_iter(&mut self, problem: &mut Problem<O>, state: I) -> Result<(I, Option<KV>), Error>;

    /// Initializes the algorithm
    ///
    /// This is executed before any iterations are performed. It can be used to perform
    /// precomputations. The default implementation corresponds to doing nothing.
    fn init(&mut self, _problem: &mut Problem<O>, state: I) -> Result<(I, Option<KV>), Error> {
        Ok((state, None))
    }

    /// Checks whether basic termination reasons apply.
    ///
    /// Terminate if
    ///
    /// 1) algorithm was terminated somewhere else in the Executor
    /// 2) iteration count exceeds maximum number of iterations
    /// 3) cost is lower than target cost
    ///
    /// This can be overwritten in a `Solver` implementation; however it is not advised.
    fn terminate_internal(&mut self, state: &I) -> TerminationReason {
        let solver_terminate = self.terminate(state);
        if solver_terminate.terminated() {
            return solver_terminate;
        }
        if state.get_iter() >= state.get_max_iters() {
            return TerminationReason::MaxItersReached;
        }
        if state.get_cost() <= state.get_target_cost() {
            return TerminationReason::TargetCostReached;
        }
        TerminationReason::NotTerminated
    }

    /// Checks whether the algorithm must be terminated
    fn terminate(&mut self, _state: &I) -> TerminationReason {
        TerminationReason::NotTerminated
    }
}
