// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

extern crate argmin;
use argmin::prelude::*;
use argmin::{ArgminProblem, NelderMead};
use argmin::testfunctions::rosenbrock;

fn run() -> Result<(), Box<std::error::Error>> {
    // Define cost function
    let cost = |x: &Vec<f64>| -> f64 { rosenbrock(x, 1_f64, 100_f64) };

    // Set up problem
    let mut prob = ArgminProblem::new(&cost);
    prob.target_cost(0.01);

    // Set up GradientDecent solver
    let mut solver = NelderMead::new();
    solver.max_iters(100);

    // Choose the starting points.
    let init_params = vec![vec![0.0, 0.1], vec![2.0, 1.5], vec![2.0, -1.0]];

    let result = solver.run(&prob, &init_params)?;

    println!("{:?}", result);
    Ok(())
}

fn main() {
    if let Err(ref e) = run() {
        println!("error: {}", e);
    }
}
