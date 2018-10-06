// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

#![feature(custom_attribute)]
#![feature(unrestricted_attribute_tokens)]
#![allow(unused_attributes)]

#[macro_use]
extern crate argmin;
#[macro_use]
extern crate argmin_codegen;
use argmin::prelude::*;
use argmin::solver::trustregion::*;
use argmin::testfunctions::{rosenbrock_2d, rosenbrock_2d_derivative, rosenbrock_2d_hessian};

fn rosenbrock(x: &Vec<f64>) -> f64 {
    rosenbrock_2d(x, 1.0, 100.0)
}

fn rosenbrock_gradient(x: &Vec<f64>) -> Vec<f64> {
    rosenbrock_2d_derivative(x, 1.0, 100.0)
}

fn rosenbrock_hessian(x: &Vec<f64>) -> Vec<Vec<f64>> {
    let bla = rosenbrock_2d_hessian(x, 1.0, 100.0);
    // hacky...
    let mut out = vec![vec![0.0_f64; 2]; 2];
    out[0][0] = bla[0];
    out[0][1] = bla[1];
    out[1][0] = bla[2];
    out[1][1] = bla[3];
    out
}

#[derive(Clone, ArgminOperator)]
#[output_type(f64)]
#[parameters_type(Vec<f64>)]
#[hessian_type(Vec<Vec<f64>>)]
#[cost_function(rosenbrock)]
#[gradient(rosenbrock_gradient)]
#[hessian(rosenbrock_hessian)]
struct MyProblem {}

fn run() -> Result<(), Error> {
    // Define cost function
    let cost = MyProblem {};

    // definie inital parameter vector
    let init_param: Vec<f64> = vec![1.2, 1.2];
    // let init_param: Vec<f64> = vec![-1.2, 1.0];

    let iters = 1000;
    let mut solver = TrustRegion::new(Box::new(cost), init_param);
    solver.set_max_iters(iters);
    solver.add_logger(ArgminSlogLogger::term_noblock());

    solver.run()?;

    // Wait a second (lets the logger flush everything before printing to screen again)
    std::thread::sleep(std::time::Duration::from_secs(1));
    println!("{:?}", solver.result());
    Ok(())
}

fn main() {
    if let Err(ref e) = run() {
        println!("{} {}", e.as_fail(), e.backtrace());
        std::process::exit(1);
    }
}
