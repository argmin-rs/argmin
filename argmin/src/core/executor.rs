// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

// TODO: Logging of "initial info"

#[cfg(feature = "serde1")]
use crate::core::{
    serialization::load_checkpoint, ArgminCheckpoint, CheckpointMode, DeserializeOwnedAlias,
};
use crate::core::{
    ArgminKV, ArgminResult, Error, Observe, Observer, ObserverMode, OpWrapper, Solver, State,
    TerminationReason,
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
    pub op: OpWrapper<O>,
    /// State
    pub(crate) state: Option<I>,
    /// Storage for observers
    #[cfg_attr(feature = "serde1", serde(skip))]
    observers: Observer<I>,
    /// Checkpoint
    #[cfg(feature = "serde1")]
    checkpoint: ArgminCheckpoint,
    /// Indicates whether Ctrl-C functionality should be active or not
    ctrlc: bool,
    /// Indicates whether to time execution or not
    timer: bool,
}

impl<O, S, I> Executor<O, S, I>
where
    S: Solver<I>,
    I: State<Operator = O>,
{
    /// Create a new executor with a `solver` and an initial parameter `init_param`
    pub fn new(op: O, solver: S) -> Self {
        let state = Some(I::new());
        Executor {
            solver,
            op: OpWrapper::new(op),
            state,
            observers: Observer::new(),
            #[cfg(feature = "serde1")]
            checkpoint: ArgminCheckpoint::default(),
            ctrlc: true,
            timer: true,
        }
    }

    /// Create a new executor from a checkpoint
    #[cfg(feature = "serde1")]
    pub fn from_checkpoint<P: AsRef<Path>>(path: P, op: O) -> Result<Self, Error>
    where
        Self: Sized + DeserializeOwnedAlias,
    {
        let (mut executor, state): (Self, I) = load_checkpoint(path)?;
        executor.state = Some(state);
        executor.op = OpWrapper::new(op);
        Ok(executor)
    }

    /// Run the executor
    pub fn run(mut self) -> Result<ArgminResult<I>, Error> {
        let total_time = if self.timer {
            Some(instant::Instant::now())
        } else {
            None
        };

        let mut state = self.state.take().unwrap();

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

        // let mut op_wrapper = OpWrapper::new(&self.op);
        let init_data = self.solver.init(&mut self.op, &mut state)?;

        // If init() returned something, deal with it
        if let Some(data) = &init_data {
            state.update(data);
        }

        if !self.observers.is_empty() {
            let mut logs = make_kv!("max_iters" => state.get_max_iters(););

            if let Some(data) = init_data {
                logs = logs.merge(&mut data.get_kv());
            }

            // Observe after init
            self.observers.observe_init(S::NAME, &logs)?;
        }

        state.set_func_counts(&self.op);

        while running.load(Ordering::SeqCst) {
            // check first if it has already terminated
            // This should probably be solved better.
            // First, check if it isn't already terminated. If it isn't, evaluate the
            // stopping criteria. If `self.terminate()` is called without the checking
            // whether it has terminated already, then it may overwrite a termination set
            // within `next_iter()`!
            if !state.terminated() {
                state.termination_reason(self.solver.terminate_internal(&state));
            }
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

            let data = self.solver.next_iter(&mut self.op, &mut state)?;

            state.set_func_counts(&self.op);

            // End time measurement
            let duration = if self.timer {
                Some(start.unwrap().elapsed())
            } else {
                None
            };

            state.update(&data);

            if !self.observers.is_empty() {
                let mut log = data.get_kv();

                if self.timer {
                    let duration = duration.unwrap();
                    log = log.merge(&mut make_kv!(
                        "time" => duration.as_secs() as f64 + f64::from(duration.subsec_nanos()) * 1e-9;
                    ));
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
            state.termination_reason(TerminationReason::Aborted);
        }
        Ok(ArgminResult::new(self.op, state))
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

    /// Set maximum number of iterations
    #[must_use]
    pub fn max_iters(mut self, iters: u64) -> Self {
        self.state.as_mut().unwrap().max_iters(iters);
        self
    }

    /// Set target cost value
    #[must_use]
    pub fn target_cost(mut self, cost: I::Float) -> Self {
        self.state.as_mut().unwrap().target_cost(cost);
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
    use crate::core::{ArgminIterData, ArgminOp, IterState, MinimalNoOperator, State};
    use approx::assert_relative_eq;

    #[test]
    fn test_update() {
        #[derive(Clone)]
        #[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
        struct TestSolver {}

        impl<O> Solver<IterState<O>> for TestSolver
        where
            O: ArgminOp,
        {
            fn next_iter(
                &mut self,
                _op: &mut OpWrapper<O>,
                _state: &mut IterState<O>,
            ) -> Result<ArgminIterData<IterState<O>>, Error> {
                Ok(ArgminIterData::new())
            }
        }

        let op = MinimalNoOperator::new();
        let solver = TestSolver {};

        let mut executor =
            Executor::new(op, solver).configure(|config| config.param(vec![0.0, 0.0]));

        // 1) Parameter vector changes, but not cost (continues to be `Inf`)
        let new_param = vec![1.0, 1.0];
        let new_iterdata: ArgminIterData<IterState<MinimalNoOperator>> =
            ArgminIterData::new().param(new_param.clone());
        executor.state.as_mut().unwrap().update(&new_iterdata);
        assert_eq!(
            executor.state.as_ref().unwrap().get_best_param().unwrap(),
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
        let new_iterdata: ArgminIterData<IterState<MinimalNoOperator>> = ArgminIterData::new()
            .param(new_param.clone())
            .cost(new_cost);
        executor.state.as_mut().unwrap().update(&new_iterdata);
        assert_eq!(
            executor.state.as_ref().unwrap().get_best_param().unwrap(),
            new_param
        );
        assert_relative_eq!(
            executor.state.as_ref().unwrap().get_best_cost(),
            new_cost,
            epsilon = f64::EPSILON
        );

        // 3) Parameter vector and cost changes to something worse
        let old_param = executor.state.as_ref().unwrap().get_best_param();
        let new_param = vec![3.0, 3.0];
        let old_cost = executor.state.as_ref().unwrap().get_best_cost();
        let new_cost = old_cost + 1.0;
        let new_iterdata: ArgminIterData<IterState<MinimalNoOperator>> =
            ArgminIterData::new().param(new_param).cost(new_cost);
        executor.state.as_mut().unwrap().update(&new_iterdata);
        assert_eq!(executor.state.as_ref().unwrap().get_best_param(), old_param);
        assert_relative_eq!(
            executor.state.as_ref().unwrap().get_best_cost(),
            old_cost,
            epsilon = f64::EPSILON
        );

        // 4) `-Inf` is better than `Inf`
        let solver = TestSolver {};
        let mut executor =
            Executor::new(op, solver).configure(|config| config.param(vec![0.0, 0.0]));

        let new_param = vec![1.0, 1.0];
        let new_cost = std::f64::NEG_INFINITY;
        let new_iterdata: ArgminIterData<IterState<MinimalNoOperator>> = ArgminIterData::new()
            .param(new_param.clone())
            .cost(new_cost);
        executor.state.as_mut().unwrap().update(&new_iterdata);
        assert_eq!(
            executor.state.as_ref().unwrap().get_best_param().unwrap(),
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
        let old_param = executor.state.as_ref().unwrap().get_best_param().unwrap();
        let new_param = vec![6.0, 6.0];
        let new_cost = std::f64::INFINITY;
        let new_iterdata: ArgminIterData<IterState<MinimalNoOperator>> =
            ArgminIterData::new().param(new_param).cost(new_cost);
        executor.state.as_mut().unwrap().update(&new_iterdata);
        assert_eq!(
            executor.state.as_ref().unwrap().get_best_param().unwrap(),
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
