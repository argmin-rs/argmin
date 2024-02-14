// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use argmin::{
    core::{observers::ObserverMode, CostFunction, Error, Executor},
    solver::simulatedannealing::{Anneal, SATempFunc, SimulatedAnnealing},
};
use argmin_observer_spectator::SpectatorBuilder;
use argmin_testfunctions::rosenbrock;
use rand::{distributions::Uniform, prelude::*};
use rand_xoshiro::Xoshiro256PlusPlus;
use std::sync::{Arc, Mutex};

struct Rosenbrock {
    /// lower bound
    lower_bound: Vec<f64>,
    /// upper bound
    upper_bound: Vec<f64>,
    /// Random number generator. We use a `Arc<Mutex<_>>` here because `Anneal` requires
    /// `self` to be passed as an immutable reference. This gives us thread safe interior
    /// mutability.
    rng: Arc<Mutex<Xoshiro256PlusPlus>>,
}

impl Rosenbrock {
    /// Constructor
    pub fn new(lower_bound: Vec<f64>, upper_bound: Vec<f64>) -> Self {
        Rosenbrock {
            lower_bound,
            upper_bound,
            rng: Arc::new(Mutex::new(Xoshiro256PlusPlus::from_entropy())),
        }
    }
}

impl CostFunction for Rosenbrock {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        Ok(rosenbrock(param))
    }
}

impl Anneal for Rosenbrock {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Float = f64;

    /// Anneal a parameter vector
    fn anneal(&self, param: &Vec<f64>, temp: f64) -> Result<Vec<f64>, Error> {
        let mut param_n = param.clone();
        let mut rng = self.rng.lock().unwrap();
        let distr = Uniform::from(0..param.len());
        for _ in 0..(temp.floor() as u64 + 1) {
            let idx = rng.sample(distr);
            let val = rng.sample(Uniform::new_inclusive(-0.0001, 0.0001));
            param_n[idx] += val;
            param_n[idx] = param_n[idx].clamp(self.lower_bound[idx], self.upper_bound[idx]);
        }
        Ok(param_n)
    }
}

fn run() -> Result<(), Error> {
    let num = 10;
    // Define bounds
    let lower_bound: Vec<f64> = vec![-5.0; num];
    let upper_bound: Vec<f64> = vec![5.0; num];

    // Define cost function
    let operator = Rosenbrock::new(lower_bound, upper_bound);

    // Define initial parameter vector
    let init_param: Vec<f64> = vec![-0.9; num];

    // Define initial temperature
    let temp = 1000.0;

    // Set up simulated annealing solver
    let solver = SimulatedAnnealing::new(temp)?.with_temp_func(SATempFunc::Boltzmann);

    let spectator = SpectatorBuilder::new()
        // .with_name("name_your_run")
        .select(&["cost", "best_cost", "t"])
        .build();

    /////////////////////////
    // Run solver          //
    /////////////////////////
    let res = Executor::new(operator, solver)
        .configure(|state| {
            state
                .param(init_param)
                .max_iters(1_000_000)
                .target_cost(0.0)
        })
        // Add spectator observer
        .add_observer(spectator, ObserverMode::Always)
        .run()?;

    // Print result
    println!("{res}");
    Ok(())
}

fn main() {
    if let Err(ref e) = run() {
        println!("{e}");
        std::process::exit(1);
    }
}
