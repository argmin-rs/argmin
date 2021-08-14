// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

// TODO: Logging of "initial info"

use crate::core::serialization::*;
use crate::core::{
    ArgminCheckpoint, ArgminIterData, ArgminKV, ArgminOp, ArgminResult, Error, IterState, Observe,
    Observer, ObserverMode, OpWrapper, Solver, TerminationReason,
};
use instant;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Executes a solver
#[derive(Clone, Serialize, Deserialize)]
pub struct Executor<O: ArgminOp, S> {
    /// solver
    solver: S,
    /// operator
    #[serde(skip)]
    pub op: OpWrapper<O>,
    /// State
    #[serde(bound = "IterState<O>: Serialize")]
    state: IterState<O>,
    /// Storage for observers
    #[serde(skip)]
    observers: Observer<O>,
    /// Checkpoint
    checkpoint: ArgminCheckpoint,
    /// Indicates whether Ctrl-C functionality should be active or not
    ctrlc: bool,
    /// Indicates whether to time execution or not
    timer: bool,
}

impl<O, S> Executor<O, S>
where
    O: ArgminOp,
    S: Solver<O>,
{
    /// Create a new executor with a `solver` and an initial parameter `init_param`
    pub fn new(op: O, solver: S, init_param: O::Param) -> Self {
        let state = IterState::new(init_param);
        Executor {
            solver,
            op: OpWrapper::new(op),
            state,
            observers: Observer::new(),
            checkpoint: ArgminCheckpoint::default(),
            ctrlc: true,
            timer: true,
        }
    }

    /// Create a new executor from a checkpoint
    pub fn from_checkpoint<P: AsRef<Path>>(path: P, op: O) -> Result<Self, Error>
    where
        Self: Sized + DeserializeOwned,
    {
        let mut executor: Self = load_checkpoint(path)?;
        executor.op = OpWrapper::new(op);
        Ok(executor)
        // load_checkpoint(path)
    }

    fn update(&mut self, data: &ArgminIterData<O>) -> Result<(), Error> {
        if let Some(cur_param) = data.get_param() {
            self.state.param(cur_param);
        }
        if let Some(cur_cost) = data.get_cost() {
            self.state.cost(cur_cost);
        }
        // check if parameters are the best so far
        if self.state.get_cost() <= self.state.get_best_cost() {
            let param = self.state.get_param();
            let cost = self.state.get_cost();
            self.state.best_param(param).best_cost(cost);
            self.state.new_best();
        }

        if let Some(grad) = data.get_grad() {
            self.state.grad(grad);
        }
        if let Some(hessian) = data.get_hessian() {
            self.state.hessian(hessian);
        }
        if let Some(jacobian) = data.get_jacobian() {
            self.state.jacobian(jacobian);
        }
        if let Some(population) = data.get_population() {
            self.state.population(population.clone());
        }

        if let Some(termination_reason) = data.get_termination_reason() {
            self.state.termination_reason(termination_reason);
        }
        Ok(())
    }

    /// Run the executor
    pub fn run(mut self) -> Result<ArgminResult<O>, Error> {
        let total_time = if self.timer {
            Some(instant::Instant::now())
        } else {
            None
        };

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
        let init_data = self.solver.init(&mut self.op, &self.state)?;

        // If init() returned something, deal with it
        if let Some(data) = &init_data {
            self.update(data)?;
        }

        if !self.observers.is_empty() {
            let mut logs = make_kv!("max_iters" => self.state.get_max_iters(););

            if let Some(data) = init_data {
                logs = logs.merge(&mut data.get_kv());
            }

            // Observe after init
            self.observers.observe_init(S::NAME, &logs)?;
        }

        self.state.set_func_counts(&self.op);

        while running.load(Ordering::SeqCst) {
            // check first if it has already terminated
            // This should probably be solved better.
            // First, check if it isn't already terminated. If it isn't, evaluate the
            // stopping criteria. If `self.terminate()` is called without the checking
            // whether it has terminated already, then it may overwrite a termination set
            // within `next_iter()`!
            if !self.state.terminated() {
                self.state
                    .termination_reason(self.solver.terminate_internal(&self.state));
            }
            // Now check once more if the algorithm has terminated. If yes, then break.
            if self.state.terminated() {
                break;
            }

            // Start time measurement
            let start = if self.timer {
                Some(instant::Instant::now())
            } else {
                None
            };

            let data = self.solver.next_iter(&mut self.op, &self.state)?;

            self.state.set_func_counts(&self.op);

            // End time measurement
            let duration = if self.timer {
                Some(start.unwrap().elapsed())
            } else {
                None
            };

            self.update(&data)?;

            if !self.observers.is_empty() {
                let mut log = data.get_kv();

                if self.timer {
                    let duration = duration.unwrap();
                    log = log.merge(&mut make_kv!(
                        "time" => duration.as_secs() as f64 + f64::from(duration.subsec_nanos()) * 1e-9;
                    ));
                }
                self.observers.observe_iter(&self.state, &log)?;
            }

            // increment iteration number
            self.state.increment_iter();

            self.checkpoint.store_cond(&self, self.state.get_iter())?;

            if self.timer {
                total_time.map(|total_time| self.state.time(Some(total_time.elapsed())));
            }

            // Check if termination occured inside next_iter()
            if self.state.terminated() {
                break;
            }
        }

        // in case it stopped prematurely and `termination_reason` is still `NotTerminated`,
        // someone must have pulled the handbrake
        if self.state.get_iter() < self.state.get_max_iters() && !self.state.terminated() {
            self.state.termination_reason(TerminationReason::Aborted);
        }

        Ok(ArgminResult::new(self.op.get_op(), self.state))
    }

    /// Attaches a observer which implements `ArgminLog` to the solver.
    pub fn add_observer<OBS: Observe<O> + 'static>(
        mut self,
        observer: OBS,
        mode: ObserverMode,
    ) -> Self {
        self.observers.push(observer, mode);
        self
    }

    /// Set maximum number of iterations
    pub fn max_iters(mut self, iters: u64) -> Self {
        self.state.max_iters(iters);
        self
    }

    /// Set target cost value
    pub fn target_cost(mut self, cost: O::Float) -> Self {
        self.state.target_cost(cost);
        self
    }

    /// Set cost value
    pub fn cost(mut self, cost: O::Float) -> Self {
        self.state.cost(cost);
        self
    }

    /// Set Gradient
    pub fn grad(mut self, grad: O::Param) -> Self {
        self.state.grad(grad);
        self
    }

    /// Set Hessian
    pub fn hessian(mut self, hessian: O::Hessian) -> Self {
        self.state.hessian(hessian);
        self
    }

    /// Set Jacobian
    pub fn jacobian(mut self, jacobian: O::Jacobian) -> Self {
        self.state.jacobian(jacobian);
        self
    }

    /// Set checkpoint directory
    pub fn checkpoint_dir(mut self, dir: &str) -> Self {
        self.checkpoint.set_dir(dir);
        self
    }

    /// Set checkpoint name
    pub fn checkpoint_name(mut self, dir: &str) -> Self {
        self.checkpoint.set_name(dir);
        self
    }

    /// Set the checkpoint mode
    pub fn checkpoint_mode(mut self, mode: CheckpointMode) -> Self {
        self.checkpoint.set_mode(mode);
        self
    }

    /// Turn Ctrl-C handling on or off (default: on)
    pub fn ctrlc(mut self, ctrlc: bool) -> Self {
        self.ctrlc = ctrlc;
        self
    }

    /// Turn timer on or off (default: on)
    pub fn timer(mut self, timer: bool) -> Self {
        self.timer = timer;
        self
    }
}
