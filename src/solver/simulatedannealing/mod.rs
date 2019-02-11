// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! * [Simulated Annealing](struct.SimulatedAnnealing.html)
//!
//! # References
//!
//! [0] [Wikipedia](https://en.wikipedia.org/wiki/Simulated_annealing)
//!
//! [1] S Kirkpatrick, CD Gelatt Jr, MP Vecchi. (1983). "Optimization by Simulated Annealing".
//! Science 13 May 1983, Vol. 220, Issue 4598, pp. 671-680
//! DOI: 10.1126/science.220.4598.671  

use crate::prelude::*;
use argmin_codegen::ArgminSolver;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

/// Temperature functions for Simulated Annealing.
///
/// Given the initial temperature `t_init` and the iteration number `i`, the current temperature
/// `t_i` is given as follows:
///
/// * `SATempFunc::TemperatureFast`: `t_i = t_init / i`
/// * `SATempFunc::Boltzmann`: `t_i = t_init / ln(i)`
/// * `SATempFunc::Exponential`: `t_i = t_init * 0.95^i`
/// * `SATempFunc::Custom`: User provided temperature update function which must have the function
///   signature `&Fn(init_temp: f64, iteration_number: u64) -> f64`
pub enum SATempFunc {
    /// `t_i = t_init / i`
    TemperatureFast,
    /// `t_i = t_init / ln(i)`
    Boltzmann,
    /// `t_i = t_init * x^i`
    Exponential(f64),
    /// User-provided temperature function. The first parameter must be the current temperature and
    /// the second parameter must be the iteration number.
    Custom(Box<Fn(f64, u64) -> f64>),
}

impl std::default::Default for SATempFunc {
    fn default() -> Self {
        SATempFunc::Boltzmann
    }
}

/// Simulated Annealing
///
/// # Example
///
/// ```
/// extern crate argmin;
/// extern crate rand;
/// use argmin::prelude::*;
/// use argmin::solver::simulatedannealing::{SATempFunc, SimulatedAnnealing};
/// use argmin::testfunctions::rosenbrock;
/// use rand::prelude::*;
/// use std::sync::{Arc, Mutex};;
/// use serde::{Deserialize, Serialize};
///
/// #[derive(Clone, Serialize, Deserialize)]
/// struct Rosenbrock {
///     /// Parameter a, usually 1.0
///     a: f64,
///     /// Parameter b, usually 100.0
///     b: f64,
///     /// lower bound
///     lower_bound: Vec<f64>,
///     /// upper bound
///     upper_bound: Vec<f64>,
///     /// Random number generator. We use a `Arc<Mutex<_>>` here because `ArgminOperator` requires
///     /// `self` to be passed as an immutable reference. This gives us thread safe interior
///     /// mutability.
///     #[serde(skip)]
///     #[serde(default = "default_rng")]
///     rng: Arc<Mutex<SmallRng>>,
/// }
///
/// fn default_rng() -> Arc<Mutex<SmallRng>> {
///     Arc::new(Mutex::new(SmallRng::from_entropy()))
/// }
///
/// impl std::default::Default for Rosenbrock {
///     fn default() -> Self {
///         let lower_bound: Vec<f64> = vec![-5.0, -5.0];
///         let upper_bound: Vec<f64> = vec![5.0, 5.0];
///         Rosenbrock::new(1.0, 100.0, lower_bound, upper_bound)
///     }
/// }
///
/// impl Rosenbrock {
///     /// Constructor
///     pub fn new(a: f64, b: f64, lower_bound: Vec<f64>, upper_bound: Vec<f64>) -> Self {
///         Rosenbrock {
///             a,
///             b,
///             lower_bound,
///             upper_bound,
///             rng: Arc::new(Mutex::new(SmallRng::from_entropy())),
///         }
///     }
/// }
///
/// impl ArgminOp for Rosenbrock {
///     type Param= Vec<f64>;
///     type Output = f64;
///     type Hessian = ();
///
///     fn apply(&self, param: &Vec<f64>) -> Result<f64, Error> {
///         Ok(rosenbrock(param, self.a, self.b))
///     }
///
///     /// This function is called by the annealing function
///     fn modify(&self, param: &Vec<f64>, temp: f64) -> Result<Vec<f64>, Error> {
///         let mut param_n = param.clone();
///         // Perform modifications to a degree proportional to the current temperature `temp`.
///         for _ in 0..(temp.floor() as u64 + 1) {
///             // Compute random index of the parameter vector using the supplied random number
///             // generator.
///             let mut rng = self.rng.lock().unwrap();
///             let idx = (*rng).gen_range(0, param.len());
///
///             // Compute random number in [0.01, 0.01].
///             let val = 0.01 * (*rng).gen_range(-1.0, 1.0);
///
///             // modify previous parameter value at random position `idx` by `val`
///             let tmp = param[idx] + val;
///
///             // check if bounds are violated. If yes, project onto bound.
///             if tmp > self.upper_bound[idx] {
///                 param_n[idx] = self.upper_bound[idx];
///             } else if tmp < self.lower_bound[idx] {
///                 param_n[idx] = self.lower_bound[idx];
///             } else {
///                 param_n[idx] = param[idx] + val;
///             }
///         }
///         Ok(param_n)
///     }
/// }
///
/// fn run() -> Result<(), Error> {
///     // Define bounds
///     let lower_bound: Vec<f64> = vec![-5.0, -5.0];
///     let upper_bound: Vec<f64> = vec![5.0, 5.0];
///
///     // Define cost function
///     let operator = Rosenbrock::new(1.0, 100.0, lower_bound, upper_bound);
///
///     // definie inital parameter vector
///     let init_param: Vec<f64> = vec![1.0, 1.2];
///
///     // Define initial temperature
///     let temp = 15.0;
///
///     // Set up simulated annealing solver
///     let mut solver = SimulatedAnnealing::new(operator, init_param, temp)?;
///
///     // Optional: Define temperature function (defaults to `SATempFunc::TemperatureFast`)
///     solver.temp_func(SATempFunc::Boltzmann);
///
///     // Optional: Attach a logger
///     solver.add_logger(ArgminSlogLogger::term());
///
///     /////////////////////////
///     // Stopping criteria   //
///     /////////////////////////
///
///     // Optional: Set maximum number of iterations (defaults to `std::u64::MAX`)
///     solver.set_max_iters(1_000);
///
///     // Optional: Set target cost function value (defaults to `std::f64::NEG_INFINITY`)
///     solver.set_target_cost(0.0);
///
///     // Optional: stop if there was no new best solution after 100 iterations
///     solver.stall_best(100);
///
///     // Optional: stop if there was no accepted solution after 100 iterations
///     solver.stall_accepted(100);
///
///     /////////////////////////
///     // Reannealing         //
///     /////////////////////////
///
///     // Optional: Reanneal after 100 iterations (resets temperature to initial temperature)
///     solver.reannealing_fixed(100);
///
///     // Optional: Reanneal after no accepted solution has been found for 50 iterations
///     solver.reannealing_accepted(50);
///
///     // Optional: Start reannealing after no new best solution has been found for 80 iterations
///     solver.reannealing_best(80);
///
///     /////////////////////////
///     // Run solver          //
///     /////////////////////////
///
///     solver.run()?;
///
///     // Wait a second (lets the logger flush everything before printing again)
///     std::thread::sleep(std::time::Duration::from_secs(1));
///
///     // Print result
///     println!("{:?}", solver.result());
///     Ok(())
/// }
///
/// fn main() {
///     if let Err(ref e) = run() {
///         println!("{} {}", e.as_fail(), e.backtrace());
///         std::process::exit(1);
///     }
/// }
/// ```
///
/// # References
///
/// [0] [Wikipedia](https://en.wikipedia.org/wiki/Simulated_annealing)
///
/// [1] S Kirkpatrick, CD Gelatt Jr, MP Vecchi. (1983). "Optimization by Simulated Annealing".
/// Science 13 May 1983, Vol. 220, Issue 4598, pp. 671-680
/// DOI: 10.1126/science.220.4598.671  
#[derive(ArgminSolver, Serialize, Deserialize)]
#[log("initial_temperature" => "self.init_temp")]
#[log("stall_iter_accepted_limit" => "self.stall_iter_accepted_limit")]
#[log("stall_iter_best_limit" => "self.stall_iter_best_limit")]
#[log("reanneal_fixed" => "self.reanneal_fixed")]
#[log("reanneal_accepted" => "self.reanneal_accepted")]
#[log("reanneal_best" => "self.reanneal_best")]
pub struct SimulatedAnnealing<O>
where
    O: ArgminOp<Output = f64>,
{
    /// Initial temperature
    init_temp: f64,
    /// which temperature function?
    #[serde(skip)]
    temp_func: SATempFunc,
    /// Number of iterations used for the caluclation of temperature. This is needed for
    /// reannealing!
    temp_iter: u64,
    /// Iterations since the last accepted solution
    stall_iter_accepted: u64,
    /// Stop if stall_iter_accepted exceedes this number
    stall_iter_accepted_limit: u64,
    /// Iterations since the last best solution was found
    stall_iter_best: u64,
    /// Stop if stall_iter_best exceedes this number
    stall_iter_best_limit: u64,
    /// Reanneal after this number of iterations is reached
    reanneal_fixed: u64,
    /// Similar to `iter`, but will be reset to 0 when reannealing is performed
    reanneal_iter_fixed: u64,
    /// Reanneal after no accepted solution has been found for `reanneal_accepted` iterations
    reanneal_accepted: u64,
    /// Similar to `stall_iter_accepted`, but will be reset to 0 when reannealing  is performed
    reanneal_iter_accepted: u64,
    /// Reanneal after no new best solution has been found for `reanneal_best` iterations
    reanneal_best: u64,
    /// Similar to `stall_iter_best`, but will be reset to 0 when reannealing is performed
    reanneal_iter_best: u64,
    /// current temperature
    cur_temp: f64,
    /// previous cost
    prev_cost: f64,
    /// random number generator
    #[serde(skip)]
    #[serde(default = "default_rng")]
    rng: SmallRng,
    /// base
    base: ArgminBase<O>,
}

// fn default_rng() -> Arc<Mutex<SmallRng>> {
fn default_rng() -> SmallRng {
    // Arc::new(Mutex::new(SmallRng::from_entropy()))
    SmallRng::from_entropy()
}

impl<O> SimulatedAnnealing<O>
where
    O: ArgminOp<Output = f64>,
{
    /// Constructor
    ///
    /// Parameters:
    ///
    /// * `cost_function`: cost function
    /// * `init_param`: initial parameter vector
    /// * `init_temp`: initial temperature
    pub fn new(
        cost_function: O,
        init_param: <O as ArgminOp>::Param,
        init_temp: f64,
    ) -> Result<Self, Error> {
        let prev_cost = cost_function.apply(&init_param)?;
        if init_temp <= 0_f64 {
            Err(ArgminError::InvalidParameter {
                text: "initial temperature".to_string(),
            }
            .into())
        } else {
            Ok(SimulatedAnnealing {
                init_temp,
                temp_func: SATempFunc::TemperatureFast,
                temp_iter: 0u64,
                stall_iter_accepted: 0u64,
                stall_iter_accepted_limit: std::u64::MAX,
                stall_iter_best: 0u64,
                stall_iter_best_limit: std::u64::MAX,
                reanneal_fixed: std::u64::MAX,
                reanneal_iter_fixed: 0,
                reanneal_accepted: std::u64::MAX,
                reanneal_iter_accepted: 0,
                reanneal_best: std::u64::MAX,
                reanneal_iter_best: 0,
                cur_temp: init_temp,
                prev_cost,
                rng: SmallRng::from_entropy(),
                base: ArgminBase::new(cost_function, init_param),
            })
        }
    }

    /// Set temperature function to one of the options in `SATempFunc`.
    pub fn temp_func(&mut self, temperature_func: SATempFunc) -> &mut Self {
        self.temp_func = temperature_func;
        self
    }

    /// The optimization stops after there has been no accepted solution after `iter` iterations
    pub fn stall_accepted(&mut self, iter: u64) -> &mut Self {
        self.stall_iter_accepted_limit = iter;
        self
    }

    /// The optimization stops after there has been no new best solution after `iter` iterations
    pub fn stall_best(&mut self, iter: u64) -> &mut Self {
        self.stall_iter_best_limit = iter;
        self
    }

    /// Start reannealing after `iter` iterations
    pub fn reannealing_fixed(&mut self, iter: u64) -> &mut Self {
        self.reanneal_fixed = iter;
        self
    }

    /// Start reannealing after no accepted solution has been found for `iter` iterations
    pub fn reannealing_accepted(&mut self, iter: u64) -> &mut Self {
        self.reanneal_accepted = iter;
        self
    }

    /// Start reannealing after no new best solution has been found for `iter` iterations
    pub fn reannealing_best(&mut self, iter: u64) -> &mut Self {
        self.reanneal_best = iter;
        self
    }

    /// Acceptance function
    ///
    /// Any solution which satisfies `next_cost < prev_cost` will be accepted. Solutions worse than
    /// the previous one are accepted with a probability given as:
    ///
    /// `1 / (1 + exp((next_cost - prev_cost) / current_temperature))`,
    ///
    /// which will always be between 0 and 0.5.
    fn accept(&mut self, next_param: &<O as ArgminOp>::Param, next_cost: f64) -> (bool, bool) {
        let prob: f64 = self.rng.gen();
        let mut new_best = false;
        let accepted = if (next_cost < self.prev_cost)
            || (1.0 / (1.0 + ((next_cost - self.prev_cost) / self.cur_temp).exp()) > prob)
        {
            // If yes, update the parameter vector for the next iteration.
            self.prev_cost = next_cost;
            self.set_cur_param(next_param.clone());

            // In case the new solution is better than the current best, update best as well.
            if next_cost < self.best_cost() {
                new_best = true;
                self.set_best_cost(next_cost);
                self.set_best_param(next_param.clone());
            }
            true
        } else {
            false
        };
        (accepted, new_best)
    }

    /// Update the temperature based on the current iteration number.
    ///
    /// Updates are performed based on specific update functions. See `SATempFunc` for details.
    fn update_temperature(&mut self) {
        self.cur_temp = match self.temp_func {
            SATempFunc::TemperatureFast => self.init_temp / ((self.temp_iter + 1) as f64),
            SATempFunc::Boltzmann => self.init_temp / ((self.temp_iter + 1) as f64).ln(),
            SATempFunc::Exponential(x) => self.init_temp * x.powf((self.temp_iter + 1) as f64),
            SATempFunc::Custom(ref func) => func(self.init_temp, self.temp_iter),
        };
    }

    /// Perform annealing
    fn anneal(&mut self) -> Result<<O as ArgminOp>::Param, Error> {
        let tmp = self.cur_param();
        let cur_temp = self.cur_temp;
        self.modify(&tmp, cur_temp)
    }

    /// Perform reannealing
    fn reanneal(&mut self) -> (bool, bool, bool) {
        let out = (
            self.reanneal_iter_fixed >= self.reanneal_fixed,
            self.reanneal_iter_accepted >= self.reanneal_accepted,
            self.reanneal_iter_best >= self.reanneal_best,
        );
        if out.0 || out.1 || out.2 {
            self.reanneal_iter_fixed = 0;
            self.reanneal_iter_accepted = 0;
            self.reanneal_iter_best = 0;
            self.cur_temp = self.init_temp;
            self.temp_iter = 0;
        }
        out
    }

    /// Update the stall iter variables
    fn update_stall_and_reanneal_iter(&mut self, accepted: bool, new_best: bool) {
        self.stall_iter_accepted = if accepted {
            0
        } else {
            self.stall_iter_accepted + 1
        };

        self.reanneal_iter_accepted = if accepted {
            0
        } else {
            self.reanneal_iter_accepted + 1
        };

        self.stall_iter_best = if new_best {
            0
        } else {
            self.stall_iter_best + 1
        };

        self.reanneal_iter_best = if new_best {
            0
        } else {
            self.reanneal_iter_best + 1
        };
    }
}

impl<O> ArgminIter for SimulatedAnnealing<O>
where
    O: ArgminOp<Output = f64>,
{
    type Param = <O as ArgminOp>::Param;
    type Output = <O as ArgminOp>::Output;
    type Hessian = <O as ArgminOp>::Hessian;

    /// Perform one iteration of SA algorithm
    fn next_iter(&mut self) -> Result<ArgminIterData<Self::Param>, Error> {
        // Careful: The order in here is *very* important, even if it may not seem so. Everything
        // is linked to the iteration number, and getting things mixed up will lead to strange
        // behaviour. None of these strange behaviour is dangerous, but still.

        // Make a move
        let new_param = self.anneal()?;

        // Evaluate cost function with new parameter vector
        let new_cost = self.apply(&new_param)?;

        // Decide whether new parameter vector should be accepted.
        // If no, move on with old parameter vector.
        let (accepted, new_best) = self.accept(&new_param, new_cost);

        // Update stall iter variables
        self.update_stall_and_reanneal_iter(accepted, new_best);

        let (r_fixed, r_accepted, r_best) = self.reanneal();

        // Update temperature for next iteration.
        self.temp_iter += 1;
        // Todo: this variable may not be necessary (temp_iter does the same?)
        self.reanneal_iter_fixed += 1;

        self.update_temperature();

        let mut out = ArgminIterData::new(new_param, new_cost);
        out.add_kv(make_kv!(
            "t" => self.cur_temp;
            "new_be" => new_best;
            "acc" => accepted;
            "st_i_be" => self.stall_iter_best;
            "st_i_ac" => self.stall_iter_accepted;
            "ra_i_fi" => self.reanneal_iter_fixed;
            "ra_i_be" => self.reanneal_iter_best;
            "ra_i_ac" => self.reanneal_iter_accepted;
            "ra_fi" => r_fixed;
            "ra_be" => r_best;
            "ra_ac" => r_accepted;
        ));
        Ok(out)
    }
}
