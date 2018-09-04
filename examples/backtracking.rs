// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

#[macro_use]
extern crate argmin;
extern crate rand;
use argmin::prelude::*;
// use argmin_core::WriteToFile;
use argmin::solver::linesearch::*;
use argmin::testfunctions::{sphere, sphere_derivative};

#[derive(Clone)]
struct MyProblem {}

impl ArgminOperator for MyProblem {
    type Parameters = Vec<f64>;
    type OperatorOutput = f64;

    fn apply(&self, param: &Vec<f64>) -> Result<f64, Error> {
        Ok(sphere(param))
    }

    fn gradient(&self, param: &Vec<f64>) -> Result<Vec<f64>, Error> {
        Ok(sphere_derivative(param))
    }

    box_clone!();
}

fn run() -> Result<(), Error> {
    // definie inital parameter vector
    let init_param: Vec<f64> = vec![1.0, 0.0];

    let operator = MyProblem {};

    // Set up Line Search method
    let iters = 100;
    let mut solver = BacktrackingLineSearch::new(Box::new(operator));
    // BacktrackingLineSearch::new(Box::new(operator), vec![-1.0, 0.0], init_param, 0.9)?;

    solver.set_search_direction(vec![-1.0, 0.0]);
    solver.set_initial_parameter(init_param);
    solver.set_rho(0.9)?;
    solver.calc_inital_cost()?;
    solver.calc_inital_gradient()?;
    solver.set_initial_alpha(1.0)?;
    solver.set_max_iters(iters);
    // solver.add_writer(WriteToFile::new());
    solver.add_logger(ArgminSlogLogger::term());
    solver.add_logger(ArgminSlogLogger::file("file.log")?);
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
