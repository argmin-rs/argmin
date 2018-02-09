// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

extern crate argmin;
extern crate ndarray;
use ndarray::Array1;
use argmin::BacktrackingLineSearch;
use argmin::testfunctions::{rosenbrock_derivative_nd, rosenbrock_nd};
use argmin::ArgminSolver;

fn run() -> Result<(), Box<std::error::Error>> {
    // Define cost function
    let cost = |x: &Array1<f64>| -> f64 { rosenbrock_nd(x, 1_f64, 100_f64) };
    let gradient = |x: &Array1<f64>| -> Array1<f64> { rosenbrock_derivative_nd(x, 1_f64, 100_f64) };

    // Set up GradientDecent solver
    let mut solver = BacktrackingLineSearch::new(&cost, &gradient);
    // solver.max_iters(10_000);

    let x = Array1::from_vec(vec![4.1, 3.0]);
    let p = gradient(&x);

    let result = solver.run(-p, &x)?;

    // print result
    println!("{:?}", result);
    Ok(())
}

fn main() {
    if let Err(ref e) = run() {
        println!("error: {}", e);
    }
}
