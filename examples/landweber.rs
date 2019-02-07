// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

extern crate argmin;
use argmin::prelude::*;
use argmin::solver::landweber::*;
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
    // define inital parameter vector
    let init_param: Vec<f64> = vec![1.2, 1.2];
    let operator = Rosenbrock {};

    let iters = 4000;
    let mut solver = Landweber::new(operator, 0.001, init_param)?;
    solver.set_max_iters(iters);
    solver.add_logger(ArgminSlogLogger::term());
    solver.run()?;

    // Wait a second (lets the logger flush everything before printing to screen again)
    std::thread::sleep(std::time::Duration::from_secs(1));
    println!("{}", solver.result());
    Ok(())
}

fn main() {
    if let Err(ref e) = run() {
        println!("{} {}", e.as_fail(), e.backtrace());
    }
}
