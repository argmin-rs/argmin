// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

extern crate argmin;
extern crate ndarray;
use ndarray::{Array1, Array2};
use argmin::prelude::*;
use argmin::{ArgminProblem, Newton};
use argmin::testfunctions::{rosenbrock_derivative_nd, rosenbrock_hessian_nd, rosenbrock_nd};

fn run() -> Result<(), Box<std::error::Error>> {
    // Define cost function
    // Choose either `Rosenbrock` or `Sphere` function.
    let cost = |x: &Array1<f64>| -> f64 { rosenbrock_nd(x, 1_f64, 100_f64) };
    let gradient = |x: &Array1<f64>| -> Array1<f64> { rosenbrock_derivative_nd(x, 1_f64, 100_f64) };
    let hessian = |x: &Array1<f64>| -> Array2<f64> { rosenbrock_hessian_nd(x, 1_f64, 100_f64) };

    // Set up problem
    // The problem requires a cost function, gradient and hessian.
    let mut prob = ArgminProblem::new(&cost);
    prob.gradient(&gradient);
    prob.hessian(&hessian);

    // Set up Newton solver
    let mut solver = Newton::new();
    solver.max_iters(10);

    // define inital parameter vector
    let init_param: Array1<f64> = Array1::from_vec(vec![1.5, 1.5]);
    println!("{:?}", init_param);

    // Manually solve it
    solver.init(&prob, &init_param)?;

    let mut par;
    loop {
        par = solver.next_iter()?;
        println!("{:?}", par);
        if solver.terminate() {
            break;
        };
    }

    // run it from scratch using the `run` method
    solver.init(&prob, &init_param)?;
    par = solver.run(&prob, &init_param)?;
    println!("{:?}", par);

    Ok(())
}

fn main() {
    if let Err(ref e) = run() {
        println!("error: {}", e);
    }
}
