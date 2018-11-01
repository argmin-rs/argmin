// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

extern crate argmin;
extern crate argmin_testfunctions;
extern crate rand;
use argmin::solver::conjugategradient::*;
use argmin_testfunctions::{rosenbrock_2d, rosenbrock_2d_derivative};
// use argmin_testfunctions::{sphere, sphere_derivative};
// use argmin_core::WriteToFile;
use argmin::prelude::*;

#[derive(Clone)]
struct MyProblem {}

impl ArgminOperator for MyProblem {
    type Parameters = Vec<f64>;
    type OperatorOutput = f64;
    type Hessian = ();

    fn apply(&self, p: &Vec<f64>) -> Result<f64, Error> {
        Ok(rosenbrock_2d(p, 1.0, 100.0))
    }

    fn gradient(&self, p: &Vec<f64>) -> Result<Vec<f64>, Error> {
        Ok(rosenbrock_2d_derivative(p, 1.0, 100.0))
    }

    box_clone!();
}

fn run() -> Result<(), Error> {
    // definie inital parameter vector
    let init_param: Vec<f64> = vec![1.2, 1.2];
    let operator = MyProblem {};

    // Set up Conjugate Gradient method
    let iters = 20;
    let mut solver = NonlinearConjugateGradient::new_pr(Box::new(operator), init_param)?;
    solver.set_max_iters(iters);
    solver.set_target_cost(0.0);
    // solver.add_writer(WriteToFile::new());
    solver.add_logger(ArgminSlogLogger::term());
    // solver.add_logger(ArgminSlogLogger::file("file.log")?);
    solver.run()?;

    // Wait a second (lets the logger flush everything before printing to screen again)
    std::thread::sleep(std::time::Duration::from_secs(1));
    println!("{:?}", solver.result());
    Ok(())
}

fn main() {
    if let Err(ref e) = run() {
        println!("{} {}", e.as_fail(), e.backtrace());
    }
}
