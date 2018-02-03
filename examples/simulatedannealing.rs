// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

extern crate argmin;
use argmin::ArgminSolver;
use argmin::problem::Problem;
use argmin::sa::{SATempFunc, SimulatedAnnealing};
use argmin::testfunctions::rosenbrock;

fn run() -> Result<(), Box<std::error::Error>> {
    // Define cost function
    let cost = |x: &Vec<f64>| rosenbrock(x, 1_f64, 100_f64);

    // Define bounds
    let lower_bound: Vec<f64> = vec![-1.5, -0.5];
    let upper_bound: Vec<f64> = vec![2.0, 3.0];

    // Set up problem
    let mut prob: Problem<_, _, f64> = Problem::new(&cost);
    prob.bounds(&lower_bound, &upper_bound);
    prob.target_cost(0.01);

    // Set up simulated annealing solver
    let mut solver = SimulatedAnnealing::new(10.0, 1_000_000_000)?;
    solver.temp_func(SATempFunc::Exponential(0.8));

    // definie inital parameter vector
    let init_param: Vec<f64> = vec![0.0, 0.0];
    // let init_param: Vec<f64> = prob.random_param()?;

    let result = solver.run(&prob, &init_param)?;

    // print result
    println!("{:?}", result);
    Ok(())
}

fn main() {
    if let Err(ref e) = run() {
        println!("error: {}", e);
    }
}
