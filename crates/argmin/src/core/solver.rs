// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::{Error, Problem, State, TerminationReason, TerminationStatus, KV};

/// The interface all solvers are required to implement.
///
/// Every solver needs to implement this trait in order to function with the `Executor`.
/// It handles initialization ([`init`](`Solver::init`)), each iteration of the solver
/// ([`next_iter`](`Solver::next_iter`)), and termination of the algorithm
/// ([`terminate`](`Solver::terminate`) and [`terminate_internal`](`Solver::terminate_internal`)).
/// Only `next_iter` is mandatory to implement, all others provide default implementations.
///
/// A `Solver` should be (de)serializable in order to work with checkpointing.
///
/// # Example
///
/// ```
/// use argmin::core::{
///     ArgminFloat, Solver, IterState, CostFunction, Error, KV, Problem, TerminationReason, TerminationStatus
/// };
///
/// #[derive(Clone)]
/// struct OptimizationAlgorithm {}
///
/// impl<O, P, G, J, H, R, F> Solver<O, IterState<P, G, J, H, R, F>> for OptimizationAlgorithm
/// where
///     O: CostFunction<Param = P, Output = F>,
///     P: Clone,
///     F: ArgminFloat
/// {
///     const NAME: &'static str = "OptimizationAlgorithm";
///
///     fn init(
///         &mut self,
///         problem: &mut Problem<O>,
///         state: IterState<P, G, J, H, R, F>,
///     ) -> Result<(IterState<P, G, J, H, R, F>, Option<KV>), Error> {
///         // Initialize algorithm, update `state`.
///         // Implementing this method is optional.
///         Ok((state, None))
///     }
///
///     fn next_iter(
///         &mut self,
///         problem: &mut Problem<O>,
///         state: IterState<P, G, J, H, R, F>,
///     ) -> Result<(IterState<P, G, J, H, R, F>, Option<KV>), Error> {
///         // Compute single iteration of algorithm, update `state`.
///         // Implementing this method is required.
///         Ok((state, None))
///     }
///     
///     fn terminate(&mut self, state: &IterState<P, G, J, H, R, F>) -> TerminationStatus {
///         // Check if stopping criteria are met.
///         // Implementing this method is optional.
///         TerminationStatus::NotTerminated
///     }
/// }
/// ```
pub trait Solver<O, I: State> {
    /// Name of the solver. Mainly used in [Observers](`crate::core::observers::Observe`).
    const NAME: &'static str;

    /// Initializes the algorithm.
    ///
    /// Executed before any iterations are performed and has access to the optimization problem
    /// definition and the internal state of the solver.
    /// Returns an updated `state` and optionally a `KV` which holds key-value pairs used in
    /// [Observers](`crate::core::observers::Observe`).
    /// The default implementation returns the unaltered `state` and no `KV`.
    fn init(&mut self, _problem: &mut Problem<O>, state: I) -> Result<(I, Option<KV>), Error> {
        Ok((state, None))
    }

    /// Computes a single iteration of the algorithm and has access to the optimization problem
    /// definition and the internal state of the solver.
    /// Returns an updated `state` and optionally a `KV` which holds key-value pairs used in
    /// [Observers](`crate::core::observers::Observe`).
    fn next_iter(&mut self, problem: &mut Problem<O>, state: I) -> Result<(I, Option<KV>), Error>;

    /// Checks whether basic termination reasons apply.
    ///
    /// Terminate if
    ///
    /// 1) algorithm was terminated somewhere else in the Executor
    /// 2) iteration count exceeds maximum number of iterations
    /// 3) best cost is lower than or equal to the target cost
    ///
    /// This can be overwritten; however it is not advised. It is recommended to implement other
    /// stopping criteria via ([`terminate`](`Solver::terminate`).
    fn terminate_internal(&mut self, state: &I) -> TerminationStatus {
        let solver_status = self.terminate(state);
        if solver_status.terminated() {
            return solver_status;
        }
        if state.get_iter() >= state.get_max_iters() {
            return TerminationStatus::Terminated(TerminationReason::MaxItersReached);
        }
        if state.get_best_cost() <= state.get_target_cost() {
            return TerminationStatus::Terminated(TerminationReason::TargetCostReached);
        }
        TerminationStatus::NotTerminated
    }

    /// Used to implement stopping criteria, in particular criteria which are not covered by
    /// ([`terminate_internal`](`Solver::terminate_internal`).
    ///
    /// This method has access to the internal state and returns an `TerminationReason`.
    fn terminate(&mut self, _state: &I) -> TerminationStatus {
        TerminationStatus::NotTerminated
    }
}
