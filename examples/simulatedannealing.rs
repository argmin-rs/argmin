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
use rand::Rng;
// use rand::ThreadRng;

#[derive(Clone)]
struct MyProblem {
    lower_bound: Vec<f64>,
    upper_bound: Vec<f64>,
    // rng: ThreadRng,
}

impl MyProblem {
    pub fn new(lower_bound: Vec<f64>, upper_bound: Vec<f64>) -> Self {
        MyProblem {
            lower_bound,
            upper_bound,
            // rng: rand::thread_rng(),
        }
    }
}

impl ArgminOperator for MyProblem {
    type Parameters = Vec<f64>;
    type OperatorOutput = f64;
    type Hessian = ();

    fn apply(&self, param: &Vec<f64>) -> Result<f64, Error> {
        Ok(rosenbrock(param, 1.0, 100.0))
    }

    fn modify(&self, param: &Vec<f64>, temp: f64) -> Result<Vec<f64>, Error> {
        let mut param_n = param.clone();
        let mut rng = rand::thread_rng();
        for _ in 0..(temp.floor() as u64 + 1) {
            // let idx = self.rng.gen_range(0, param.len());
            let idx = rng.gen_range(0, param.len());
            // let val = 0.001 * self.rng.gen_range(-1.0, 1.0);
            let val = 0.001 * rng.gen_range(-1.0, 1.0);
            let tmp = param[idx] + val;
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
    let operator = MyProblem::new(lower_bound, upper_bound);

    // definie inital parameter vector
    let init_param: Vec<f64> = vec![1.0, 1.2];

    // Set up simulated annealing solver
    // let iters = 10_000;
    // let iters = 31;
    let iters = 10;
    let temp = 0.5;
    // solver.temp_func(SATempFunc::Exponential(0.8));
    let mut solver = SimulatedAnnealing::new(&operator, init_param, temp)?;
    solver.set_max_iters(iters);
    solver.set_target_cost(0.0);
    solver.reannealing_fixed(10);
    solver.temp_func(SATempFunc::Boltzmann);
    solver.add_logger(ArgminSlogLogger::term());
    solver.add_logger(ArgminSlogLogger::file("file.log")?);
    // solver.add_writer(WriteToFile::new());

    // .stall_best(100);
    solver.run()?;

    // Wait a second (lets the logger flush everything before printing to screen again)
    std::thread::sleep(std::time::Duration::from_secs(1));
    println!("{:?}", solver.result());
    Ok(())
}

fn main() {
    if let Err(ref e) = run() {
        println!("{} {}", e.as_fail(), e.backtrace());
        std::process::exit(1);
    }
}
