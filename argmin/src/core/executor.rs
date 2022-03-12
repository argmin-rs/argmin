// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

#[cfg(feature = "serde1")]
use crate::core::{load_checkpoint, Checkpoint, CheckpointMode};
use crate::core::{
    DeserializeOwnedAlias, Error, Observe, Observer, ObserverMode, OptimizationResult, Problem,
    SerializeAlias, Solver, State, TerminationReason, KV,
};
use instant;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "serde1")]
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Executes a solver
#[derive(Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct Executor<O, S, I> {
    /// solver
    solver: S,
    /// operator
    #[cfg_attr(feature = "serde1", serde(skip))]
    pub problem: Problem<O>,
    /// State
    pub(crate) state: Option<I>,
    /// Storage for observers
    #[cfg_attr(feature = "serde1", serde(skip))]
    observers: Observer<I>,
    /// Checkpoint
    #[cfg(feature = "serde1")]
    checkpoint: Checkpoint,
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
    /// Create a new executor with a `solver` and an initial parameter `init_param`
    pub fn new(problem: O, solver: S) -> Self {
        let state = Some(I::new());
        Executor {
            solver,
            problem: Problem::new(problem),
            state,
            observers: Observer::new(),
            #[cfg(feature = "serde1")]
            checkpoint: Checkpoint::default(),
            ctrlc: true,
            timer: true,
        }
    }

    /// Create a new executor from a checkpoint
    #[cfg(feature = "serde1")]
    pub fn from_checkpoint<P: AsRef<Path>>(path: P, problem: O) -> Result<Self, Error>
    where
        Self: Sized + DeserializeOwnedAlias,
    {
        let (mut executor, state): (Self, I) = load_checkpoint(path)?;
        executor.state = Some(state);
        executor.problem = Problem::new(problem);
        Ok(executor)
    }

    /// Run the executor
    pub fn run(mut self) -> Result<OptimizationResult<O, I>, Error> {
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

            #[cfg(feature = "serde1")]
            self.checkpoint
                .store_cond(&self, &state, state.get_iter())?;

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

    /// Attaches a observer which implements `ArgminLog` to the solver.
    #[must_use]
    pub fn add_observer<OBS: Observe<I> + 'static>(
        mut self,
        observer: OBS,
        mode: ObserverMode,
    ) -> Self {
        self.observers.push(observer, mode);
        self
    }

    /// Configure the solver
    #[must_use]
    pub fn configure<F: FnOnce(I) -> I>(mut self, init: F) -> Self {
        let state = self.state.take().unwrap();
        let state = init(state);
        self.state = Some(state);
        self
    }

    /// Set checkpoint directory
    #[cfg(feature = "serde1")]
    #[must_use]
    pub fn checkpoint_dir(mut self, dir: &str) -> Self {
        self.checkpoint.set_dir(dir);
        self
    }

    /// Set checkpoint name
    #[cfg(feature = "serde1")]
    #[must_use]
    pub fn checkpoint_name(mut self, dir: &str) -> Self {
        self.checkpoint.set_name(dir);
        self
    }

    /// Set the checkpoint mode
    #[cfg(feature = "serde1")]
    #[must_use]
    pub fn checkpoint_mode(mut self, mode: CheckpointMode) -> Self {
        self.checkpoint.set_mode(mode);
        self
    }

    /// Turn Ctrl-C handling on or off (default: on)
    #[must_use]
    pub fn ctrlc(mut self, ctrlc: bool) -> Self {
        self.ctrlc = ctrlc;
        self
    }

    /// Turn timer on or off (default: on)
    #[must_use]
    pub fn timer(mut self, timer: bool) -> Self {
        self.timer = timer;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{ArgminFloat, IterState, PseudoProblem};
    use approx::assert_relative_eq;

    #[test]
    fn test_update() {
        #[derive(Clone)]
        #[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
        struct TestSolver {}

        impl<O, P, G, J, H, F> Solver<O, IterState<P, G, J, H, F>> for TestSolver
        where
            P: Clone,
            F: ArgminFloat,
        {
            fn next_iter(
                &mut self,
                _problem: &mut Problem<O>,
                state: IterState<P, G, J, H, F>,
            ) -> Result<(IterState<P, G, J, H, F>, Option<KV>), Error> {
                Ok((state, None))
            }
        }

        let problem = PseudoProblem::new();
        let solver = TestSolver {};

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
