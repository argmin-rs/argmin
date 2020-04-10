// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

extern crate argmin;
extern crate argmin_testfunctions;
use argmin::prelude::*;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin_testfunctions::{sphere, sphere_derivative};

#[derive(Clone, Default)]
struct Sphere {}

impl ArgminOp for Sphere {
    type Param = Vec<f64>;
    type Output = f64;
    type Hessian = ();
    type Jacobian = ();

    fn apply(&self, param: &Vec<f64>) -> Result<f64, Error> {
        Ok(sphere(param))
    }

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
    let mut solver = MoreThuenteLineSearch::new();

    // The following parameters do not follow the builder pattern because they are part of the
    // ArgminLineSearch trait which needs to be object safe.

    // Set search direction
    solver.set_search_direction(vec![-2.0, 0.0]);

    // Set initial step length
    solver.set_init_alpha(1.0)?;

    let init_cost = operator.apply(&init_param)?;
    let init_grad = operator.gradient(&init_param)?;

    // Run solver
    let res = Executor::new(operator, solver, init_param)
        .add_observer(ArgminSlogLogger::term(), ObserverMode::Always)
        .max_iters(10)
        // the following two are optional. If they are not provided, they will be computed
        .cost(init_cost)
        .grad(init_grad)
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
