// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use argmin::core::{CostFunction, Error, Executor, Gradient, LineSearch, ObserverMode, SlogLogger};
use argmin::solver::linesearch::HagerZhangLineSearch;
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
    // Define inital parameter vector
    let init_param: Vec<f64> = vec![1.0, 0.0];

    // Problem definition
    let operator = Sphere {};

    // Set up line search method
    let mut solver = HagerZhangLineSearch::new();

    // The following parameters do not follow the builder pattern because they are part of the
    // LineSearch trait which needs to be object safe.

    // Set search direction
    solver.set_search_direction(vec![-1.5, 0.0]);

    // Set initial step length
    solver.set_init_alpha(1.0)?;

    let init_cost = operator.cost(&init_param)?;
    let init_grad = operator.gradient(&init_param)?;

    // Run solver
    let res = Executor::new(operator, solver)
        .add_observer(SlogLogger::term(), ObserverMode::Always)
        // Gradient and cost are optional. If they are not provided, they will be computed
        .configure(|config| {
            config
                .param(init_param)
                .grad(init_grad)
                .cost(init_cost)
                .max_iters(10)
        })
        .run()?;

    // Wait a second (lets the logger flush everything before printing again)
    std::thread::sleep(std::time::Duration::from_secs(1));

    // Print Result
    println!("{}", res);
    Ok(())
}

fn main() {
    if let Err(ref e) = run() {
        println!("{}", e);
    }
}
