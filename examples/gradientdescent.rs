// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

extern crate argmin;
extern crate ndarray;
use ndarray::Array1;
use argmin::prelude::*;
use argmin::{BacktrackingLineSearch, GDGammaUpdate, GradientDescent, Problem};
use argmin::testfunctions::{rosenbrock_derivative_nd, rosenbrock_nd};

fn run() -> Result<(), Box<std::error::Error>> {
    // Define cost function
    let cost = |x: &Array1<f64>| -> f64 { rosenbrock_nd(x, 1_f64, 100_f64) };
    let gradient = |x: &Array1<f64>| -> Array1<f64> { rosenbrock_derivative_nd(x, 1_f64, 100_f64) };

    // Set up problem
    // The problem requires a cost function, lower and upper bounds and takes an optional gradient.
    let mut prob = Problem::new(&cost);
    prob.gradient(&gradient);

    // Set up GradientDecent solver
    let mut solver = GradientDescent::new();
    // Set the maximum number of iterations to 10000
    solver.max_iters(10_000);
    // Choose the method which calculates the step width. `GDGammaUpdate::Constant(0.0001)` sets a
    // constant step width of 0.0001 while `GDGammaUpdate::BarzilaiBorwein` updates the step width
    // according to TODO
    // solver.gamma_update(GDGammaUpdate::Constant(0.0001));
    solver.gamma_update(GDGammaUpdate::BarzilaiBorwein);

    // define inital parameter vector
    let init_param: Array1<f64> = Array1::from_vec(vec![1.5, 1.5]);
    println!("{:?}", init_param);

    // Actually run the solver on the problem.
    let result1 = solver.run(&prob, &init_param)?;

    // print result
    println!("{:?}", result1);

    // Define new solver
    let mut solver = GradientDescent::new();
    solver.max_iters(10_000);

    // Initialize a backtracking line search method to use for the gamma update method
    let mut linesearch = BacktrackingLineSearch::new(&cost, &gradient);
    linesearch.alpha(1.0);
    solver.gamma_update(GDGammaUpdate::BacktrackingLineSearch(linesearch));

    // Run solver
    let result2 = solver.run(&prob, &init_param)?;

    // print result
    println!("{:?}", result2);

    // Manually solve it
    // `GradientDescent` also allows you to initialize the solver yourself and run each iteration
    // manually. This is particularly useful if you need to get intermediate results or if your
    // need to print out information that is otherwise not accessible to you.
    let mut solver = GradientDescent::new();
    solver.init(&prob, &init_param)?;

    let mut par;
    loop {
        par = solver.next_iter()?;
        // println!("{:?}", par);
        if par.iters >= result1.iters {
            break;
        };
    }

    println!("{:?}", par);

    Ok(())
}

fn main() {
    if let Err(ref e) = run() {
        println!("error: {}", e);
    }
}
