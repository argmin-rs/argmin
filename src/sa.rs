// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! TODO
//!
//! * [ ] Different acceptance functions
//! * [ ] Early stopping criterions

use rand;
use rand::distributions::{IndependentSample, Range};
use errors::*;
use prelude::*;
use problem::ArgminProblem;
use result::ArgminResult;
use termination::TerminationReason;

/// Definition of build in temperature functions for Simulated Annealing.
///
/// Given the initial temperature `t_init` and the iteration number `i`, the current temperature
/// `t_i` is given as follows:
///
/// `SATempFunc::TemperatureFast`: `t_i = t_init / i`
/// `SATempFunc::Boltzmann`: `t_i = t_init / ln(i)`
/// `SATempFunc::Exponential`: `t_i = t_init * 0.95^i`
/// `SATempFunc::Custom`: User provided temperature update function which has to implement the
/// function signature `&Fn(init_temp: f64, iteration_number: u64) -> f64`. See
/// `SimulatedAnnealing::custom_temp_func()` for details on how to provide a custom temperature
/// update function.
pub enum SATempFunc {
    /// `t_i = t_init / i`
    TemperatureFast,
    /// `t_i = t_init / ln(i)`
    Boltzmann,
    /// `t_i = t_init * x^i`
    Exponential(f64),
    /// User-provided temperature function. See `SimulatedAnnealing::custom_temp_func()` for
    /// details.
    Custom,
}

/// Simulated Annealing struct
pub struct SimulatedAnnealing<'a, T, U, V = U>
where
    T: ArgminParameter + 'a,
    U: ArgminCostValue + 'a,
    V: 'a,
{
    /// Initial temperature
    pub init_temp: f64,
    /// Maximum number of iterations
    pub max_iters: u64,
    /// which temperature function?
    pub temp_func: SATempFunc,
    /// Custom temperature function
    pub custom_temp_func: Option<&'a Fn(f64, u64) -> f64>,
    /// Current state of solver
    pub state: Option<SimulatedAnnealingState<'a, T, U, V>>,
}

/// State of the simulated annealing solver
pub struct SimulatedAnnealingState<'a, T, U, V>
where
    T: ArgminParameter + 'a,
    U: ArgminCostValue + 'a,
    V: 'a,
{
    /// Reference to the problem.
    problem: &'a ArgminProblem<'a, T, U, V>,
    /// Current number of iteration
    param: T,
    /// Current number of iteration
    iter: u64,
    /// current temperature
    cur_temp: f64,
    /// previous cost
    prev_cost: U,
    /// best parameter
    best_param: T,
    /// corresponding best cost
    best_cost: U,
}

impl<'a, T, U, V> SimulatedAnnealing<'a, T, U, V>
where
    T: ArgminParameter,
    U: ArgminCostValue,
    V: 'a,
{
    /// Constructor
    ///
    /// Returns an `SimulatedAnnealing` struct where all entries of the struct are set according to
    /// the parameters provided, apart from  `temp_func` and `custom_temp_func` which are set to
    /// default values (`SATempFunc::TemperatureFast` and `None`, respectively).
    ///
    /// Parameters:
    ///
    /// `problem`: problem definition
    /// `init_tmep`: Initial temperature
    /// `max_iters`: Maximum number of iterations
    pub fn new(init_temp: f64, max_iters: u64) -> Result<Self> {
        if init_temp <= 0_f64 {
            Err(
                ErrorKind::InvalidParameter("SimulatedAnnealing: Temperature must be > 0.".into())
                    .into(),
            )
        } else {
            Ok(SimulatedAnnealing {
                init_temp,
                max_iters,
                temp_func: SATempFunc::TemperatureFast,
                custom_temp_func: None,
                state: None,
            })
        }
    }

    /// Change temperature function to one of the options in `SATempFunc`.
    ///
    /// This will overwrite any custom temperature functions provided by `custom_temp_func()`.
    pub fn temp_func(&mut self, temperature_func: SATempFunc) -> &mut Self {
        self.temp_func = temperature_func;
        self
    }

    /// Provide a custom temperature function.
    ///
    /// The function has to implement the function signature `&Fn(init_temp: f64, iteration_number:
    /// u64) -> f64` and return the current temperature.
    /// This will overwrite any changes done by a call to `temp_func()`.
    pub fn custom_temp_func(&mut self, func: &'a Fn(f64, u64) -> f64) -> &mut Self {
        self.temp_func = SATempFunc::Custom;
        self.custom_temp_func = Some(func);
        self
    }

    /// Acceptance function
    ///
    /// Any solution where `next_cost < prev_cost` will be accepted. Whenever a new solution is
    /// worse than the previous one, the acceptance probability is calculated as:
    ///
    /// `1 / (1 + exp((next_cost - prev_cost) / current_temperature))`,
    ///
    /// which will always be between 0 and 0.5.
    fn accept(&self, state: &SimulatedAnnealingState<T, U, V>, next_cost: f64) -> bool {
        let prev_cost = state.prev_cost.to_f64().unwrap();
        let step = Range::new(0.0, 1.0);
        let mut rng = rand::thread_rng();
        let prob: f64 = step.ind_sample(&mut rng);
        (next_cost < prev_cost)
            || (1_f64 / (1_f64 + ((next_cost - prev_cost) / state.cur_temp).exp()) > prob)
    }
    /// Update the temperature based on the current iteration number.
    ///
    /// Updates are performed based on specific update functions. See `SATempFunc` for details.
    fn update_temperature(&self, iter: u64) -> Result<f64> {
        match self.temp_func {
            SATempFunc::TemperatureFast => Ok(self.init_temp / ((iter + 1) as f64)),
            SATempFunc::Boltzmann => Ok(self.init_temp / ((iter + 1) as f64).ln()),
            SATempFunc::Exponential(x) => if x < 1_f64 && x > 0_f64 {
                Ok(self.init_temp * x.powf((iter + 1) as f64))
            } else {
                Err(ErrorKind::InvalidParameter(
                    "SimulatedAnnealing: Parameter for exponential \
                     temperature update function needs to be >0 and <1."
                        .into(),
                ).into())
            },
            SATempFunc::Custom => match self.custom_temp_func {
                Some(func) => Ok(func(self.init_temp, iter)),
                None => Err(ErrorKind::InvalidParameter(
                    "SimulatedAnnealing: No custom temperature update function provided.".into(),
                ).into()),
            },
        }
    }
}

impl<'a, T, U, V> ArgminSolver<'a> for SimulatedAnnealing<'a, T, U, V>
where
    T: ArgminParameter + 'a,
    U: ArgminCostValue + 'a,
    V: 'a,
{
    type Parameter = T;
    type CostValue = U;
    type Hessian = V;
    type StartingPoints = T;
    type ProblemDefinition = &'a ArgminProblem<'a, Self::Parameter, Self::CostValue, Self::Hessian>;

    /// Initialize with a given problem and a starting point
    fn init(
        &mut self,
        problem: Self::ProblemDefinition,
        init_param: &Self::StartingPoints,
    ) -> Result<()> {
        let prev_cost = (problem.cost_function)(init_param);
        self.state = Some(SimulatedAnnealingState {
            problem,
            param: init_param.to_owned(),
            iter: 0_u64,
            cur_temp: self.init_temp,
            prev_cost,
            best_param: init_param.to_owned(),
            best_cost: prev_cost,
        });
        Ok(())
    }

    /// Compute next point
    fn next_iter(&mut self) -> Result<ArgminResult<Self::Parameter, Self::CostValue>> {
        // Taking the state avoids fights with the borrow checker.
        let mut state = self.state.take().unwrap();

        // initialize with an already modified parameter vector (we want at least one modification
        // anyways)
        let mut param_new = state.param.modify().0;
        for _ in 0..(state.cur_temp.floor() as u64) {
            param_new = param_new.modify().0;
            param_new = match (&state.problem.lower_bound, &state.problem.upper_bound) {
                (&Some(ref l), &Some(ref u)) => {
                    let (mut tmp, idx) = param_new.modify();
                    if tmp[idx] < l[idx] {
                        tmp[idx] = l[idx].clone();
                    }
                    if tmp[idx] > u[idx] {
                        tmp[idx] = u[idx].clone();
                    }
                    tmp
                }
                _ => param_new.modify().0,
            }
        }

        // Evaluate cost function with new parameter vector
        let new_cost = (state.problem.cost_function)(&param_new);

        // Decide whether new parameter vector should be accepted.
        // If no, move on with old parameter vector.
        if self.accept(&state, new_cost.to_f64().unwrap()) {
            // If yes, update the parameter vector for the next iteration.
            state.prev_cost = new_cost;
            state.param = param_new.clone();

            // In case the new solution is better than the current best, update best as well.
            if new_cost < state.best_cost {
                state.best_cost = new_cost;
                state.best_param = param_new;
            }
        }

        // Update temperature for next iteration.
        let cur_iter = state.iter;
        state.cur_temp = self.update_temperature(cur_iter)?;
        state.iter += 1;
        let mut out = ArgminResult::new(state.param.clone(), state.best_cost, state.iter);
        self.state = Some(state);
        out.set_termination_reason(self.terminate());
        Ok(out)
    }

    /// Stopping criterions
    make_terminate!(self,
        self.state.as_ref().unwrap().iter >= self.max_iters, TerminationReason::MaxItersReached;
        self.state.as_ref().unwrap().best_cost <= self.state.as_ref().unwrap().problem.target_cost, TerminationReason::TargetCostReached;
    );

    /// Run simulated annealing solver on problem `problem` with initial parameter `init_param`.
    make_run!(
        Self::ProblemDefinition,
        Self::StartingPoints,
        Self::Parameter,
        Self::CostValue
    );
}

unsafe impl<'a, T, U, V> Send for SimulatedAnnealing<'a, T, U, V>
where
    T: ArgminParameter + 'a,
    U: ArgminCostValue + 'a,
    V: 'a,
{
}
unsafe impl<'a, T, U, V> Sync for SimulatedAnnealing<'a, T, U, V>
where
    T: ArgminParameter + 'a,
    U: ArgminCostValue + 'a,
    V: 'a,
{
}
