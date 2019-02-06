// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

extern crate argmin;
use argmin::prelude::*;
use argmin::solver::conjugategradient::ConjugateGradient;

#[derive(Clone)]
struct MyProblem {}

impl ArgminOperator for MyProblem {
    type Parameters = Vec<f64>;
    type OperatorOutput = Vec<f64>;
    type Hessian = ();

    fn apply(&self, p: &Vec<f64>) -> Result<Vec<f64>, Error> {
        Ok(vec![4.0 * p[0] + 1.0 * p[1], 1.0 * p[0] + 3.0 * p[1]])
    }
}

fn run() -> Result<(), Error> {
    // Define inital parameter vector
    let init_param: Vec<f64> = vec![2.0, 1.0];

    // Define the right hand side `b` of `A * x = b`
    let b = vec![1.0, 2.0];

    // Set up operator
    let operator = MyProblem {};

    // Set up the solver
    let mut solver = ConjugateGradient::new(&operator, b, init_param)?;

    // Set maximum number of iterations
    solver.set_max_iters(2);

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
