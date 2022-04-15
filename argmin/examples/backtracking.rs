// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use argmin::core::observers::{ObserverMode, SlogLogger};
use argmin::core::{CostFunction, Error, Executor, Gradient, LineSearch};
use argmin::solver::linesearch::{condition::ArmijoCondition, BacktrackingLineSearch};
use argmin_testfunctions::{sphere, sphere_derivative};

struct Sphere {}

impl CostFunction for Sphere {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Vec<f64>) -> Result<f64, Error> {
        Ok(sphere(param))
    }
}

impl Gradient for Sphere {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, param: &Vec<f64>) -> Result<Vec<f64>, Error> {
        Ok(sphere_derivative(param))
    }
}

fn run() -> Result<(), Error> {
    // definie inital parameter vector
    let init_param: Vec<f64> = vec![0.7, 0.0];

    // Define problem
    let operator = Sphere {};

    // Set condition
    let cond = ArmijoCondition::new(0.5)?;

    // Set up Line Search method
    let mut solver = BacktrackingLineSearch::new(cond).rho(0.9)?;

    // The following parameters do not follow the builder pattern because they are part of the
    // ArgminLineSearch trait which needs to be object safe.

    // Set search direction
    solver.search_direction(vec![-1.0, 0.0]);

    // Set initial position
    solver.initial_step_length(1.0)?;

    let init_cost = operator.cost(&init_param)?;
    let init_grad = operator.gradient(&init_param)?;

    // Run solver
    let res = Executor::new(operator, solver)
        .add_observer(SlogLogger::term(), ObserverMode::Always)
        // Gradient and cost are optional. If they are not provided, they will be computed
        .configure(|state| {
            state
                .param(init_param)
                .grad(init_grad)
                .cost(init_cost)
                .max_iters(10)
        })
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
    }
}
