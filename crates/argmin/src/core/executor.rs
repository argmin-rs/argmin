// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::checkpointing::Checkpoint;
use crate::core::observers::{Observe, ObserverMode, Observers};
use crate::core::{
    Error, OptimizationResult, Problem, Solver, State, TerminationReason, TerminationStatus, KV,
};
use instant;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Solves an optimization problem with a solver
pub struct Executor<O, S, I> {
    /// Solver
    solver: S,
    /// Problem
    problem: Problem<O>,
    /// State
    state: Option<I>,
    /// Storage for observers
    observers: Observers<I>,
    /// Checkpoint
    checkpoint: Option<Box<dyn Checkpoint<S, I>>>,
    /// Timeout
    timeout: Option<std::time::Duration>,
    /// Indicates whether Ctrl-C functionality should be active or not
    ctrlc: bool,
    /// Indicates whether to time execution or not
    timer: bool,
}

impl<O, S, I> Executor<O, S, I>
where
    S: Solver<O, I>,
    I: State,
{
    /// Constructs an `Executor` from a user defined problem and a solver.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::Executor;
    /// # use argmin::core::test_utils::{TestSolver, TestProblem};
    /// #
    /// # type Rosenbrock = TestProblem;
    /// # type Newton = TestSolver;
    /// #
    /// // Construct an instance of the desired solver
    /// let solver = Newton::new();
    ///
    /// // `Rosenbrock` implements `CostFunction` and `Gradient` as required by the
    /// // `SteepestDescent` solver
    /// let problem = Rosenbrock {};
    ///
    /// // Create instance of `Executor` with `problem` and `solver`
    /// let executor = Executor::new(problem, solver);
    /// ```
    pub fn new(problem: O, solver: S) -> Self {
        let state = Some(I::new());
        Executor {
            solver,
            problem: Problem::new(problem),
            state,
            observers: Observers::new(),
            checkpoint: None,
            timeout: None,
            ctrlc: true,
            timer: true,
        }
    }

    /// This method gives mutable access to the internal state of the solver. This allows for
    /// initializing the state before running the `Executor`. The options for initialization depend
    /// on the type of state used by the chosen solver. Common types of state are
    /// [`IterState`](`crate::core::IterState`),
    /// [`PopulationState`](`crate::core::PopulationState`), and
    /// [`LinearProgramState`](`crate::core::LinearProgramState`). Please see the documentation of
    /// the desired solver for information about which state is used.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::Executor;
    /// # use argmin::core::test_utils::{TestSolver, TestProblem};
    /// #
    /// #  let solver = TestSolver::new();
    /// #  let problem = TestProblem::new();
    /// #  let init_param = vec![1.0f64, 0.0];
    /// #
    /// // Create instance of `Executor` with `problem` and `solver`
    /// let executor = Executor::new(problem, solver)
    ///     // Configure and initialize internal state.
    ///     .configure(|state| state.param(init_param).max_iters(10));
    /// ```
    #[must_use]
    pub fn configure<F: FnOnce(I) -> I>(mut self, init: F) -> Self {
        let state = self.state.take().unwrap();
        let state = init(state);
        self.state = Some(state);
        self
    }

    /// Runs the executor by applying the solver to the optimization problem.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{Error, Executor};
    /// # use argmin::core::test_utils::{TestSolver, TestProblem};
    /// #
    /// # fn main() -> Result<(), Error> {
    /// # let solver = TestSolver::new();
    /// # let problem = TestProblem::new();
    /// #
    /// # let init_param = vec![1.0f64, 0.0];
    /// #
    /// // Create instance of `Executor` with `problem` and `solver`
    /// let result = Executor::new(problem, solver)
    ///     // Configure and initialize internal state.
    ///     .configure(|state| state.param(init_param).max_iters(100))
    /// #   .configure(|state| state.max_iters(1))
    ///     // Execute solver
    ///     .run()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn run(mut self) -> Result<OptimizationResult<O, S, I>, Error> {
        // First, load checkpoint if given.
        if let Some(checkpoint) = self.checkpoint.as_ref() {
            if let Some((solver, state)) = checkpoint.load()? {
                self.state = Some(state);
                self.solver = solver;
            }
        }
        let total_time = if self.timer {
            Some(instant::Instant::now())
        } else {
            None
        };

        let state = self.state.take().unwrap();

        let interrupt = Arc::new(AtomicBool::new(false));

        if self.ctrlc {
            #[cfg(feature = "ctrlc")]
            {
                // Set up the Ctrl-C handler
                let interp = interrupt.clone();
                // This is currently a hack to allow checkpoints to be run again within the
                // same program (usually not really a use case anyway). Unfortunately, this
                // means that any subsequent run started afterwards will not have Ctrl-C
                // handling available... This should also be a problem in case one tries to run
                // two consecutive optimizations. There is ongoing work in the ctrlc crate
                // (channels and such) which may solve this problem. So far, we have to live
                // with this.
                let handler = move || {
                    interp.store(true, Ordering::SeqCst);
                };
                match ctrlc::set_handler(handler) {
                    Err(ctrlc::Error::MultipleHandlers) => Ok(()),
                    interp => interp,
                }?;
            }
        }

        // Only call `init` of `solver` if the current iteration number is 0. This avoids that
        // `init` is called when starting from a checkpoint (because `init` could change the state
        // of the `solver`, which would overwrite the state restored from the checkpoint).
        let mut state = if state.get_iter() == 0 {
            let (mut state, kv) = self.solver.init(&mut self.problem, state)?;
            state.update();

            if !self.observers.is_empty() {
                let kv = kv.unwrap_or(kv![]);

                // Observe after init
                self.observers
                    .observe_init(self.solver.name(), &state, &kv)?;
            }

            state.func_counts(&self.problem);
            state
        } else {
            state
        };

        while !interrupt.load(Ordering::SeqCst) {
            // check first if it has already terminated
            // This should probably be solved better.
            // First, check if it isn't already terminated. If it isn't, evaluate the
            // stopping criteria. If `self.terminate()` is called without the checking
            // whether it has terminated already, then it may overwrite a termination set
            // within `next_iter()`!
            state = if !state.terminated() {
                let term = self.solver.terminate_internal(&state);
                if let TerminationStatus::Terminated(reason) = term {
                    state.terminate_with(reason)
                } else {
                    state
                }
            } else {
                state
            };
            // Now check once more if the algorithm has terminated. If yes, then break.
            if state.terminated() {
                break;
            }

            // Start time measurement
            let start = if self.timer {
                Some(instant::Instant::now())
            } else {
                None
            };

            let (state_t, kv) = self.solver.next_iter(&mut self.problem, state)?;
            state = state_t;

            state.func_counts(&self.problem);

            // End time measurement
            let duration = if self.timer {
                Some(start.unwrap().elapsed())
            } else {
                None
            };

            state.update();

            if !self.observers.is_empty() {
                let mut log = if let Some(kv) = kv { kv } else { KV::new() };

                if self.timer {
                    let duration = duration.unwrap();
                    let tmp = kv!(
                        "time" => duration.as_secs_f64();
                    );
                    log = log.merge(tmp);
                }
                self.observers.observe_iter(&state, &log)?;
            }

            // increment iteration number
            state.increment_iter();

            if let Some(checkpoint) = self.checkpoint.as_ref() {
                checkpoint.save_cond(&self.solver, &state, state.get_iter())?;
            }

            if self.timer {
                // Increase accumulated total_time
                total_time.map(|total_time| state.time(Some(total_time.elapsed())));

                // If a timeout is set, check if timeout is reached
                if let (Some(timeout), Some(total_time)) = (self.timeout, total_time) {
                    if total_time.elapsed() > timeout {
                        state = state.terminate_with(TerminationReason::Timeout);
                    }
                }
            }

            // Check if termination occurred in the meantime
            if state.terminated() {
                break;
            }
        }

        if interrupt.load(Ordering::SeqCst) {
            // Solver execution has been interrupted manually
            state = state.terminate_with(TerminationReason::Interrupt);
        }

        if !self.observers.is_empty() {
            self.observers.observe_final(&state)?;
        }

        Ok(OptimizationResult::new(self.problem, self.solver, state))
    }

    /// Adds an observer to the executor. Observers are required to implement the
    /// [`Observe`](`crate::core::observers::Observe`) trait.
    /// The parameter `mode` defines the conditions under which the observer will be called. See
    /// [`ObserverMode`](`crate::core::observers::ObserverMode`) for details.
    ///
    /// It is possible to add multiple observers.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{Error, Executor, observers::ObserverMode};
    /// # use argmin::core::test_utils::{TestSolver, TestProblem};
    /// # use argmin_observer_slog::SlogLogger;
    /// #
    /// # fn main() -> Result<(), Error> {
    /// # let solver = TestSolver::new();
    /// # let problem = TestProblem::new();
    /// #
    /// // Create instance of `Executor` with `problem` and `solver`
    /// let executor = Executor::new(problem, solver)
    ///     .add_observer(SlogLogger::term(), ObserverMode::Always);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn add_observer<OBS: Observe<I> + 'static>(
        mut self,
        observer: OBS,
        mode: ObserverMode,
    ) -> Self {
        self.observers.push(observer, mode);
        self
    }

    /// Configures checkpointing
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{Error, Executor};
    /// # #[cfg(feature = "serde1")]
    /// # use argmin::core::checkpointing::CheckpointingFrequency;
    /// # use argmin_checkpointing_file::FileCheckpoint;
    /// # use argmin::core::test_utils::{TestSolver, TestProblem};
    /// #
    /// # fn main() -> Result<(), Error> {
    /// # let solver = TestSolver::new();
    /// # let problem = TestProblem::new();
    /// #
    /// # #[cfg(feature = "serde1")]
    /// let checkpoint = FileCheckpoint::new(
    ///     // Directory where checkpoints are saved to
    ///     ".checkpoints",
    ///     // Filename of checkpoint
    ///     "rosenbrock_optim",
    ///     // How often checkpoints should be saved
    ///     CheckpointingFrequency::Every(20)
    /// );
    ///
    /// // Create instance of `Executor` with `problem` and `solver`
    /// # #[cfg(feature = "serde1")]
    /// let executor = Executor::new(problem, solver)
    ///     // Add checkpointing
    ///     .checkpointing(checkpoint);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn checkpointing<C: 'static + Checkpoint<S, I>>(mut self, checkpoint: C) -> Self {
        self.checkpoint = Some(Box::new(checkpoint));
        self
    }

    /// Enables or disables CTRL-C handling (default: enabled). The CTRL-C handling gracefully
    /// stops the solver if it is canceled via CTRL-C (SIGINT). Requires the optional `ctrlc`
    /// feature to be set.
    ///
    /// Note that this does not work with nested `Executor`s. If a solver executes another solver
    /// internally, the inner solver needs to disable CTRL-C handling.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{Error, Executor};
    /// # use argmin::core::test_utils::{TestSolver, TestProblem};
    /// #
    /// # fn main() -> Result<(), Error> {
    /// # let solver = TestSolver::new();
    /// # let problem = TestProblem::new();
    /// #
    /// // Create instance of `Executor` with `problem` and `solver`
    /// let executor = Executor::new(problem, solver).ctrlc(false);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn ctrlc(mut self, ctrlc: bool) -> Self {
        self.ctrlc = ctrlc;
        self
    }

    /// Enables or disables timing of individual iterations (default: enabled).
    ///
    /// Setting this to false will silently be ignored in case a timeout is set.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{Error, Executor};
    /// # use argmin::core::test_utils::{TestSolver, TestProblem};
    /// #
    /// # fn main() -> Result<(), Error> {
    /// # let solver = TestSolver::new();
    /// # let problem = TestProblem::new();
    /// #
    /// // Create instance of `Executor` with `problem` and `solver`
    /// let executor = Executor::new(problem, solver).timer(false);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn timer(mut self, timer: bool) -> Self {
        if self.timeout.is_none() {
            self.timer = timer;
        }
        self
    }

    /// Sets a timeout for the run.
    ///
    /// The optimization run is stopped once the timeout is exceeded. Note that the check is
    /// performed after each iteration, therefore the actual runtime can exceed the the set
    /// duration.
    /// This also enables time measurements.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{Error, Executor};
    /// # use argmin::core::test_utils::{TestSolver, TestProblem};
    /// #
    /// # fn main() -> Result<(), Error> {
    /// # let solver = TestSolver::new();
    /// # let problem = TestProblem::new();
    /// #
    /// // Create instance of `Executor` with `problem` and `solver`
    /// let executor = Executor::new(problem, solver).timeout(std::time::Duration::from_secs(30));
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn timeout(mut self, timeout: std::time::Duration) -> Self {
        self.timer = true;
        self.timeout = Some(timeout);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::test_utils::{TestProblem, TestSolver};
    use crate::core::IterState;
    use approx::assert_relative_eq;

    #[test]
    fn test_update() {
        let problem = TestProblem::new();
        let solver = TestSolver::new();

        let mut executor = Executor::new(problem, solver).configure(
            |config: IterState<Vec<f64>, (), (), (), (), f64>| config.param(vec![0.0, 0.0]),
        );

        // 1) Parameter vector changes, but not cost (continues to be `Inf`)
        let new_param = vec![1.0, 1.0];
        executor.state = Some(executor.state.take().unwrap().param(new_param.clone()));
        executor.state.as_mut().unwrap().update();
        assert_eq!(
            *executor.state.as_ref().unwrap().get_best_param().unwrap(),
            new_param
        );
        assert!(executor
            .state
            .as_ref()
            .unwrap()
            .get_best_cost()
            .is_infinite());
        assert!(executor
            .state
            .as_ref()
            .unwrap()
            .get_best_cost()
            .is_sign_positive());

        // 2) Parameter vector and cost changes to something better
        let new_param = vec![2.0, 2.0];
        let new_cost = 10.0;
        executor.state = Some(
            executor
                .state
                .take()
                .unwrap()
                .param(new_param.clone())
                .cost(new_cost),
        );
        executor.state.as_mut().unwrap().update();
        assert_eq!(
            *executor.state.as_ref().unwrap().get_best_param().unwrap(),
            new_param
        );
        assert_relative_eq!(
            executor.state.as_ref().unwrap().get_best_cost(),
            new_cost,
            epsilon = f64::EPSILON
        );

        // 3) Parameter vector and cost changes to something worse
        let old_param = executor
            .state
            .as_ref()
            .unwrap()
            .get_best_param()
            .unwrap()
            .clone();
        let new_param = vec![3.0, 3.0];
        let old_cost = executor.state.as_ref().unwrap().get_best_cost();
        let new_cost = old_cost + 1.0;
        executor.state = Some(
            executor
                .state
                .take()
                .unwrap()
                .param(new_param)
                .cost(new_cost),
        );
        executor.state.as_mut().unwrap().update();
        assert_eq!(
            executor
                .state
                .as_ref()
                .unwrap()
                .get_best_param()
                .unwrap()
                .clone(),
            old_param
        );
        assert_relative_eq!(
            executor.state.as_ref().unwrap().get_best_cost(),
            old_cost,
            epsilon = f64::EPSILON
        );

        // 4) `-Inf` is better than `Inf`
        let solver = TestSolver {};
        let mut executor = Executor::new(problem, solver).configure(
            |config: IterState<Vec<f64>, (), (), (), (), f64>| config.param(vec![0.0, 0.0]),
        );

        let new_param = vec![1.0, 1.0];
        let new_cost = std::f64::NEG_INFINITY;
        executor.state = Some(
            executor
                .state
                .take()
                .unwrap()
                .param(new_param.clone())
                .cost(new_cost),
        );
        executor.state.as_mut().unwrap().update();
        assert_eq!(
            *executor.state.as_ref().unwrap().get_best_param().unwrap(),
            new_param
        );
        assert!(executor
            .state
            .as_ref()
            .unwrap()
            .get_best_cost()
            .is_infinite());
        assert!(executor
            .state
            .as_ref()
            .unwrap()
            .get_best_cost()
            .is_sign_negative());

        // 5) `Inf` is worse than `-Inf`
        let old_param = executor
            .state
            .as_ref()
            .unwrap()
            .get_best_param()
            .unwrap()
            .clone();
        let new_param = vec![6.0, 6.0];
        let new_cost = std::f64::INFINITY;
        executor.state = Some(
            executor
                .state
                .take()
                .unwrap()
                .param(new_param)
                .cost(new_cost),
        );
        executor.state.as_mut().unwrap().update();
        assert_eq!(
            executor
                .state
                .as_ref()
                .unwrap()
                .get_best_param()
                .unwrap()
                .clone(),
            old_param
        );
        assert!(executor
            .state
            .as_ref()
            .unwrap()
            .get_best_cost()
            .is_infinite());
        assert!(executor
            .state
            .as_ref()
            .unwrap()
            .get_best_cost()
            .is_sign_negative());
    }

    /// The solver's `init` should not be called when started from a checkpoint.
    /// See https://github.com/argmin-rs/argmin/issues/199.
    #[test]
    #[cfg(feature = "serde1")]
    fn test_checkpointing_solver_initialization() {
        use std::cell::RefCell;

        use crate::core::{
            checkpointing::CheckpointingFrequency, test_utils::TestProblem, ArgminFloat,
            CostFunction,
        };
        use serde::{Deserialize, Serialize};

        #[derive(Clone)]
        pub struct FakeCheckpoint {
            pub frequency: CheckpointingFrequency,
            pub solver: RefCell<Option<OptimizationAlgorithm>>,
            pub state: RefCell<Option<IterState<Vec<f64>, (), (), (), (), f64>>>,
        }

        impl Checkpoint<OptimizationAlgorithm, IterState<Vec<f64>, (), (), (), (), f64>>
            for FakeCheckpoint
        {
            fn save(
                &self,
                solver: &OptimizationAlgorithm,
                state: &IterState<Vec<f64>, (), (), (), (), f64>,
            ) -> Result<(), Error> {
                *self.solver.borrow_mut() = Some(solver.clone());
                *self.state.borrow_mut() = Some(state.clone());
                Ok(())
            }

            fn load(
                &self,
            ) -> Result<
                Option<(
                    OptimizationAlgorithm,
                    IterState<Vec<f64>, (), (), (), (), f64>,
                )>,
                Error,
            > {
                if self.solver.borrow().is_none() {
                    return Ok(None);
                }
                Ok(Some((
                    self.solver.borrow().clone().unwrap(),
                    self.state.borrow().clone().unwrap(),
                )))
            }

            fn frequency(&self) -> CheckpointingFrequency {
                self.frequency
            }
        }

        // Fake optimization algorithm which holds internal state which changes over time
        #[derive(Clone, Serialize, Deserialize)]
        struct OptimizationAlgorithm {
            pub internal_state: u64,
        }

        // Implement Solver for OptimizationAlgorithm
        impl<O, P, F> Solver<O, IterState<P, (), (), (), (), F>> for OptimizationAlgorithm
        where
            O: CostFunction<Param = P, Output = F>,
            P: Clone,
            F: ArgminFloat,
        {
            fn name(&self) -> &str {
                "OptimizationAlgorithm"
            }

            // Only resets internal_state to 1
            fn init(
                &mut self,
                _problem: &mut Problem<O>,
                state: IterState<P, (), (), (), (), F>,
            ) -> Result<(IterState<P, (), (), (), (), F>, Option<KV>), Error> {
                self.internal_state = 1;
                Ok((state, None))
            }

            // Increment internal_state
            fn next_iter(
                &mut self,
                _problem: &mut Problem<O>,
                state: IterState<P, (), (), (), (), F>,
            ) -> Result<(IterState<P, (), (), (), (), F>, Option<KV>), Error> {
                self.internal_state += 1;
                Ok((state, None))
            }

            // Avoid terminating early because param does not change
            fn terminate(&mut self, _state: &IterState<P, (), (), (), (), F>) -> TerminationStatus {
                TerminationStatus::NotTerminated
            }

            // Avoid terminating early because param does not change
            fn terminate_internal(
                &mut self,
                state: &IterState<P, (), (), (), (), F>,
            ) -> TerminationStatus {
                if state.get_iter() >= state.get_max_iters() {
                    TerminationStatus::Terminated(TerminationReason::MaxItersReached)
                } else {
                    TerminationStatus::NotTerminated
                }
            }
        }

        // Create random test problem
        let problem = TestProblem::new();

        // solver instance
        let solver = OptimizationAlgorithm { internal_state: 0 };

        // Create a checkpoint
        let checkpoint = FakeCheckpoint {
            frequency: CheckpointingFrequency::Always,
            solver: RefCell::new(None),
            state: RefCell::new(None),
        };

        // Create and run executor
        let executor = Executor::new(problem, solver)
            .configure(|state| state.param(vec![1.0f64, 1.0]).max_iters(10))
            .checkpointing(checkpoint.clone());

        let OptimizationResult { solver, .. } = executor.run().unwrap();

        // internal_state should be 11
        // (1 from init plus 10 iterations where it is incremented by 1)
        assert_eq!(solver.internal_state, 11);

        // Create and run solver again
        let executor = Executor::new(problem, solver)
            .configure(|state| state.param(vec![1.0f64, 1.0]).max_iters(10))
            .checkpointing(checkpoint);

        let OptimizationResult { solver, .. } = executor.run().unwrap();

        // internal_state should still be 11
        // (1 from init plus 10 iterations where it is incremented by 1)
        assert_eq!(solver.internal_state, 11);

        // Delete old checkpointing file
        let _ = std::fs::remove_file(".checkpoints/init_test.arg");
    }

    #[test]
    fn test_timeout() {
        let solver = TestSolver::new();
        let problem = TestProblem::new();
        let timeout = std::time::Duration::from_secs(2);

        let executor = Executor::new(problem, solver);
        assert!(executor.timer);
        assert!(executor.timeout.is_none());

        let executor = Executor::new(problem, solver).timer(false);
        assert!(!executor.timer);
        assert!(executor.timeout.is_none());

        let executor = Executor::new(problem, solver).timeout(timeout);
        assert!(executor.timer);
        assert_eq!(executor.timeout, Some(timeout));

        let executor = Executor::new(problem, solver).timeout(timeout).timer(false);
        assert!(executor.timer);
        assert_eq!(executor.timeout, Some(timeout));

        let executor = Executor::new(problem, solver).timer(false).timeout(timeout);
        assert!(executor.timer);
        assert_eq!(executor.timeout, Some(timeout));
    }
}
