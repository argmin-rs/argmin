// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

extern crate argmin;
extern crate rand;
use argmin::prelude::*;
use argmin::solver::simulatedannealing::{SATempFunc, SimulatedAnnealing};
use argmin::testfunctions::rosenbrock;
use rand::prelude::*;
use std::sync::Arc;
use std::sync::Mutex;

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
    rng: Arc<Mutex<SmallRng>>,
}

impl Rosenbrock {
    /// Constructor
    pub fn new(a: f64, b: f64, lower_bound: Vec<f64>, upper_bound: Vec<f64>) -> Self {
        Rosenbrock {
            a,
            b,
            lower_bound,
            upper_bound,
            rng: Arc::new(Mutex::new(SmallRng::from_entropy())),
        }
    }
}

impl ArgminOperator for Rosenbrock {
    type Parameters = Vec<f64>;
    type OperatorOutput = f64;
    type Hessian = ();

    fn apply(&self, param: &Vec<f64>) -> Result<f64, Error> {
        Ok(rosenbrock(param, self.a, self.b))
    }

    /// This function is called by the annealing function
    fn modify(&self, param: &Vec<f64>, temp: f64) -> Result<Vec<f64>, Error> {
        let mut param_n = param.clone();
        // Perform modifications to a degree proportional to the current temperature `temp`.
        for _ in 0..(temp.floor() as u64 + 1) {
            // Compute random index of the parameter vector using the supplied random number
            // generator.
            let mut rng = self.rng.lock().unwrap();
            let idx = (*rng).gen_range(0, param.len());

            // Compute random number in [0.01, 0.01].
            let val = 0.01 * (*rng).gen_range(-1.0, 1.0);

            // modify previous parameter value at random position `idx` by `val`
            let tmp = param[idx] + val;

            // check if bounds are violated. If yes, project onto bound.
            if tmp > self.upper_bound[idx] {
                param_n[idx] = self.upper_bound[idx];
            } else if tmp < self.lower_bound[idx] {
                param_n[idx] = self.lower_bound[idx];
            } else {
                param_n[idx] = param[idx] + val;
            }
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

    // definie inital parameter vector
    let init_param: Vec<f64> = vec![1.0, 1.2];

    // Define initial temperature
    let temp = 15.0;

    // Set up simulated annealing solver
    let mut solver = SimulatedAnnealing::new(&operator, init_param, temp)?;

    // Optional: Define temperature function (defaults to `SATempFunc::TemperatureFast`)
    solver.temp_func(SATempFunc::Boltzmann);

    // Optional: Attach a logger
    solver.add_logger(ArgminSlogLogger::term());

    /////////////////////////
    // Stopping criteria   //
    /////////////////////////

    // Optional: Set maximum number of iterations (defaults to `std::u64::MAX`)
    solver.set_max_iters(10_000);

    // Optional: Set target cost function value (defaults to `std::f64::NEG_INFINITY`)
    solver.set_target_cost(0.0);

    // Optional: stop if there was no new best solution after 1000 iterations
    solver.stall_best(1000);

    // Optional: stop if there was no accepted solution after 1000 iterations
    solver.stall_accepted(1000);

    /////////////////////////
    // Reannealing         //
    /////////////////////////

    // Optional: Reanneal after 1000 iterations (resets temperature to initial temperature)
    solver.reannealing_fixed(1000);

    // Optional: Reanneal after no accepted solution has been found for `iter` iterations
    solver.reannealing_accepted(500);

    // Optional: Start reannealing after no new best solution has been found for 800 iterations
    solver.reannealing_best(800);

    /////////////////////////
    // Run solver          //
    /////////////////////////

    solver.run()?;

    // Wait a second (lets the logger flush everything before printing again)
    std::thread::sleep(std::time::Duration::from_secs(1));

    // Print result
    println!("{}", solver.result());
    Ok(())
}

fn main() {
    if let Err(ref e) = run() {
        println!("{} {}", e.as_fail(), e.backtrace());
        std::process::exit(1);
    }
}
