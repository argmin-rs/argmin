// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

extern crate argmin;
use argmin::prelude::*;
use argmin::solver::conjugategradient::NonlinearConjugateGradient;
use argmin::testfunctions::{rosenbrock_2d, rosenbrock_2d_derivative};

#[derive(Clone)]
struct Rosenbrock {}

impl ArgminOperator for Rosenbrock {
    type Parameters = Vec<f64>;
    type OperatorOutput = f64;
    type Hessian = ();

    fn apply(&self, p: &Vec<f64>) -> Result<f64, Error> {
        Ok(rosenbrock_2d(p, 1.0, 100.0))
    }

    fn gradient(&self, p: &Vec<f64>) -> Result<Vec<f64>, Error> {
        Ok(rosenbrock_2d_derivative(p, 1.0, 100.0))
    }
}

fn run() -> Result<(), Error> {
    // Set up cost function
    let operator = Rosenbrock {};

    // define inital parameter vector
    let init_param: Vec<f64> = vec![1.2, 1.2];

    // Set up nonlinear conjugate gradient method
    let mut solver = NonlinearConjugateGradient::new_pr(operator, init_param)?;

    // Set maximum number of iterations
    solver.set_max_iters(20);

    // Set target cost function value
    solver.set_target_cost(0.0);

    // Set the number of iterations when a restart should be performed
    // This allows the algorithm to "forget" previous information which may not be helpful anymore.
    solver.set_restart_iters(10);

    // Set the value for the orthogonality measure.
    // Setting this parameter leads to a restart of the algorithm (setting beta = 0) after two
    // consecutive search directions are not orthogonal anymore. In other words, if this condition
    // is met:
    //
    // `|\nabla f_k^T * \nabla f_{k-1}| / | \nabla f_k ||^2 >= v`
    //
    // A typical value for `v` is 0.1.
    solver.set_restart_orthogonality(0.1);

    // Attach a logger
    solver.add_logger(ArgminSlogLogger::term());

    // Run solver
    solver.run()?;

    // Wait a second (lets the logger flush everything before printing to screen again)
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
