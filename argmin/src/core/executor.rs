// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::checkpointing::Checkpoint;
use crate::core::observers::{Observe, Observer, ObserverMode};
use crate::core::{
    DeserializeOwnedAlias, Error, OptimizationResult, Problem, SerializeAlias, Solver, State,
    TerminationReason, KV,
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
    observers: Observer<I>,
    /// Checkpoint
    checkpoint: Option<Box<dyn Checkpoint<S, I>>>,
    /// Indicates whether Ctrl-C functionality should be active or not
    ctrlc: bool,
    /// Indicates whether to time execution or not
    timer: bool,
}

impl<O, S, I> Executor<O, S, I>
where
    S: Solver<O, I>,
    I: State + SerializeAlias + DeserializeOwnedAlias,
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
            observers: Observer::new(),
            checkpoint: None,
            ctrlc: true,
            timer: true,
        }
    }

    /// This method gives mutable access to the internal state of the solver. This allows for
    /// initializing the state before running the `Executor`. The options for initialization depend
    /// on the type of state used by the chosen solver. Common types of state are
    /// [`IterState`](`crate::core::IterState`) and
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
    pub fn run(mut self) -> Result<OptimizationResult<O, I>, Error> {
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

        let running = Arc::new(AtomicBool::new(true));

        if self.ctrlc {
            #[cfg(feature = "ctrlc")]
            {
                // Set up the Ctrl-C handler
                let r = running.clone();
                // This is currently a hack to allow checkpoints to be run again within the
                // same program (usually not really a usecase anyway). Unfortunately, this
                // means that any subsequent run started afterwards will have not Ctrl-C
                // handling available... This should also be a problem in case one tries to run
                // two consecutive optimizations. There is ongoing work in the ctrlc crate
                // (channels and such) which may solve this problem. So far, we have to live
                // with this.
                match ctrlc::set_handler(move || {
                    r.store(false, Ordering::SeqCst);
                }) {
                    Err(ctrlc::Error::MultipleHandlers) => Ok(()),
                    r => r,
                }?;
            }
        }

        let (mut state, kv) = self.solver.init(&mut self.problem, state)?;
        state.update();

        if !self.observers.is_empty() {
            let mut logs = make_kv!("max_iters" => state.get_max_iters(););

            if let Some(kv) = kv {
                logs = logs.merge(kv);
            }

            // Observe after init
            self.observers.observe_init(S::NAME, &logs)?;
        }

        state.set_func_counts(&self.problem);

        while running.load(Ordering::SeqCst) {
            // check first if it has already terminated
            // This should probably be solved better.
            // First, check if it isn't already terminated. If it isn't, evaluate the
            // stopping criteria. If `self.terminate()` is called without the checking
            // whether it has terminated already, then it may overwrite a termination set
            // within `next_iter()`!
            state = if !state.terminated() {
                let term = self.solver.terminate_internal(&state);
                state.termination_reason(term)
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

            state.set_func_counts(&self.problem);

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
                    let tmp = make_kv!(
                        "time" => duration.as_secs() as f64 + f64::from(duration.subsec_nanos()) * 1e-9;
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
                total_time.map(|total_time| state.time(Some(total_time.elapsed())));
            }

            // Check if termination occured inside next_iter()
            if state.terminated() {
                break;
            }
        }

        // in case it stopped prematurely and `termination_reason` is still `NotTerminated`,
        // someone must have pulled the handbrake
        if state.get_iter() < state.get_max_iters() && !state.terminated() {
            state = state.termination_reason(TerminationReason::Aborted);
        }
        Ok(OptimizationResult::new(self.problem, state))
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
    /// # #[cfg(feature = "slog-logger")]
    /// # use argmin::core::observers::SlogLogger;
    /// #
    /// # fn main() -> Result<(), Error> {
    /// # let solver = TestSolver::new();
    /// # let problem = TestProblem::new();
    /// #
    /// // Create instance of `Executor` with `problem` and `solver`
    /// # #[cfg(feature = "slog-logger")]
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
    /// # use argmin::core::checkpointing::{FileCheckpoint, CheckpointingFrequency};
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
    /// stops the solver if it is cancelled via CTRL-C (SIGINT). Requires the optional `ctrlc`
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
        self.timer = timer;
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

        let mut executor = Executor::new(problem, solver)
            .configure(|config: IterState<Vec<f64>, (), (), (), f64>| config.param(vec![0.0, 0.0]));

        // 1) Parameter vector changes, but not cost (continues to be `Inf`)
        let new_param = vec![1.0, 1.0];
        executor.state = Some(executor.state.take().unwrap().param(new_param.clone()));
        executor.state.as_mut().unwrap().update();
        assert_eq!(
            *executor
                .state
                .as_ref()
                .unwrap()
                .get_best_param_ref()
                .unwrap(),
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
            *executor
                .state
                .as_ref()
                .unwrap()
                .get_best_param_ref()
                .unwrap(),
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
            .get_best_param_ref()
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
                .get_best_param_ref()
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
        let mut executor = Executor::new(problem, solver)
            .configure(|config: IterState<Vec<f64>, (), (), (), f64>| config.param(vec![0.0, 0.0]));

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
            *executor
                .state
                .as_ref()
                .unwrap()
                .get_best_param_ref()
                .unwrap(),
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
            .get_best_param_ref()
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
                .get_best_param_ref()
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
}
