// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use argmin::prelude::*;
use argmin::solver::simulatedannealing::{SATempFunc, SimulatedAnnealing};
use argmin_testfunctions::rosenbrock;
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use rand::distributions::Uniform;
use std::default::Default;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct Rosenbrock {
    /// Parameter a, usually 1.0
    a: f64,
    /// Parameter b, usually 100.0
    b: f64,
    /// lower bound
    lower_bound: Vec<f64>,
    /// upper bound
    upper_bound: Vec<f64>,
    /// Random number generator. We use a `Arc<Mutex<_>>` here because `ArgminOperator` requires
    /// `self` to be passed as an immutable reference. This gives us thread safe interior
    /// mutability.
    rng: Arc<Mutex<Xoshiro256PlusPlus>>,
}

impl Default for Rosenbrock {
    fn default() -> Self {
        let lower_bound: Vec<f64> = vec![-5.0, -5.0];
        let upper_bound: Vec<f64> = vec![5.0, 5.0];
        Rosenbrock::new(1.0, 100.0, lower_bound, upper_bound)
    }
}

impl Rosenbrock {
    /// Constructor
    pub fn new(a: f64, b: f64, lower_bound: Vec<f64>, upper_bound: Vec<f64>) -> Self {
        Rosenbrock {
            a,
            b,
            lower_bound,
            upper_bound,
            rng: Arc::new(Mutex::new(Xoshiro256PlusPlus::from_entropy())),
        }
    }
}

impl ArgminOp for Rosenbrock {
    type Param = Vec<f64>;
    type Output = f64;
    type Hessian = ();
    type Jacobian = ();
    type Float = f64;

    fn apply(&self, param: &Vec<f64>) -> Result<f64, Error> {
        Ok(rosenbrock(param, self.a, self.b))
    }

    /// This function is called by the annealing function
    fn modify(&self, param: &Vec<f64>, temp: f64) -> Result<Vec<f64>, Error> {
        let mut param_n = param.clone();
        let mut rng = self.rng.lock().unwrap();
        let distr = Uniform::from(0..param.len());
        // Perform modifications to a degree proportional to the current temperature `temp`.
        for _ in 0..(temp.floor() as u64 + 1) {
            // Compute random index of the parameter vector using the supplied random number
            // generator.
            let idx = rng.sample(distr);

            // Compute random number in [0.1, 0.1].
            let val = rng.sample(Uniform::new_inclusive(-0.1, 0.1));

            // modify previous parameter value at random position `idx` by `val`
            param_n[idx] += val;

            // check if bounds are violated. If yes, project onto bound.
            param_n[idx] = param_n[idx].min(self.upper_bound[idx]);
            param_n[idx] = param_n[idx].max(self.lower_bound[idx]);
        }
        Ok(param_n)
    }
}

fn run() -> Result<(), Error> {
    // Define bounds
    let lower_bound: Vec<f64> = vec![-5.0, -5.0];
    let upper_bound: Vec<f64> = vec![5.0, 5.0];

    // Define cost function
    let operator = Rosenbrock::new(1.0, 100.0, lower_bound, upper_bound);

    // Define initial parameter vector
    let init_param: Vec<f64> = vec![1.0, 1.2];

    // Define initial temperature
    let temp = 15.0;

    // Seed RNG
    let rng = Xoshiro256PlusPlus::from_entropy();

    // Set up simulated annealing solver
    let solver = SimulatedAnnealing::new(temp, rng)?
        // Optional: Define temperature function (defaults to `SATempFunc::TemperatureFast`)
        .temp_func(SATempFunc::Boltzmann)
        /////////////////////////
        // Stopping criteria   //
        /////////////////////////
        // Optional: stop if there was no new best solution after 1000 iterations
        .stall_best(1000)
        // Optional: stop if there was no accepted solution after 1000 iterations
        .stall_accepted(1000)
        /////////////////////////
        // Reannealing         //
        /////////////////////////
        // Optional: Reanneal after 1000 iterations (resets temperature to initial temperature)
        .reannealing_fixed(1000)
        // Optional: Reanneal after no accepted solution has been found for `iter` iterations
        .reannealing_accepted(500)
        // Optional: Start reannealing after no new best solution has been found for 800 iterations
        .reannealing_best(800);

    /////////////////////////
    // Run solver          //
    /////////////////////////
    let res = Executor::new(operator, solver, init_param)
        // Optional: Attach a observer
        .add_observer(ArgminSlogLogger::term(), ObserverMode::Always)
        // Optional: Set maximum number of iterations (defaults to `std::u64::MAX`)
        .max_iters(10_000)
        // Optional: Set target cost function value (defaults to `std::f64::NEG_INFINITY`)
        .target_cost(0.0)
        .run()?;

    // Wait a second (lets the logger flush everything before printing again)
    std::thread::sleep(std::time::Duration::from_secs(1));

    // Print result
    println!("{}", res);
    Ok(())
}

fn main() {
    if let Err(ref e) = run() {
        println!("{}", e);
        std::process::exit(1);
    }
}
