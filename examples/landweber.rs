// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

extern crate argmin;
extern crate rand;
use argmin::prelude::*;
use argmin::solver::landweber::*;
use argmin::testfunctions::{sphere, sphere_derivative};

#[derive(Clone)]
struct MyProblem {}

impl ArgminOperator for MyProblem {
    type Parameters = Vec<f64>;
    type OperatorOutput = f64;
    type Hessian = ();

    fn apply(&self, p: &Vec<f64>) -> Result<f64, Error> {
        Ok(sphere(p))
    }

    fn gradient(&self, p: &Vec<f64>) -> Result<Vec<f64>, Error> {
        Ok(sphere_derivative(p))
    }
}

fn run() -> Result<(), Error> {
    // definie inital parameter vector
    let init_param: Vec<f64> = vec![2.0, 1.0];
    let operator = MyProblem {};

    // Set up Conjugate Gradient method
    let iters = 20;
    let mut solver = Landweber::new(&operator, 0.1, init_param)?;
    solver.set_max_iters(iters);
    solver.add_logger(ArgminSlogLogger::term());
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
