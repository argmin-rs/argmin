// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

extern crate argmin;
use argmin::prelude::*;
use argmin::solver::linesearch::BacktrackingLineSearch;
use argmin::testfunctions::{sphere, sphere_derivative};

#[derive(Clone)]
struct Sphere {}

impl ArgminOperator for Sphere {
    type Parameters = Vec<f64>;
    type OperatorOutput = f64;
    type Hessian = ();

    fn apply(&self, param: &Vec<f64>) -> Result<f64, Error> {
        Ok(sphere(param))
    }

    fn gradient(&self, param: &Vec<f64>) -> Result<Vec<f64>, Error> {
        Ok(sphere_derivative(param))
    }
}

fn run() -> Result<(), Error> {
    // definie inital parameter vector
    let init_param: Vec<f64> = vec![1.0, 0.0];

    // Define problem
    let operator = Sphere {};

    // Set up Line Search method
    let mut solver = BacktrackingLineSearch::new(&operator);

    // Set search direction
    solver.set_search_direction(vec![-2.0, 0.0]);

    // Set initial position
    solver.set_initial_parameter(init_param);

    // Set contraction factor
    solver.set_rho(0.9)?;

    // Calculate initial cost ...
    solver.calc_initial_cost()?;

    // ... or, alternatively, set cost if it is already computed
    // solver.set_initial_cost(...);

    // Calculate initial gradient ...
    solver.calc_initial_gradient()?;
    // .. or, alternatively, set gradient if it is already computed
    // solver.set_initial_gradient(...);

    // Set initial step length
    solver.set_initial_alpha(1.0)?;

    // Set maximum number of iterations
    solver.set_max_iters(100);

    // Attach a logger
    solver.add_logger(ArgminSlogLogger::term());

    // Run solver
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
    }
}
