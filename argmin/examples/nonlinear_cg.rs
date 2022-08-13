// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use argmin::core::observers::{ObserverMode, SlogLogger};
use argmin::core::{CostFunction, Error, Executor, Gradient};
use argmin::solver::conjugategradient::{beta::PolakRibiere, NonlinearConjugateGradient};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin_testfunctions::{rosenbrock_2d, rosenbrock_2d_derivative};

struct Rosenbrock {}

impl CostFunction for Rosenbrock {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        Ok(rosenbrock_2d(p, 1.0, 100.0))
    }
}

impl Gradient for Rosenbrock {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
        Ok(rosenbrock_2d_derivative(p, 1.0, 100.0))
    }
}

fn run() -> Result<(), Error> {
    // Set up cost function
    let operator = Rosenbrock {};

    // define inital parameter vector
    let init_param: Vec<f64> = vec![1.2, 1.2];

    // set up line search
    let linesearch = MoreThuenteLineSearch::new();
    let beta_method = PolakRibiere::new();

    // Set up nonlinear conjugate gradient method
    let solver = NonlinearConjugateGradient::new(linesearch, beta_method)
        // Set the number of iterations when a restart should be performed
        // This allows the algorithm to "forget" previous information which may not be helpful anymore.
        .restart_iters(10)
        // Set the value for the orthogonality measure.
        // Setting this parameter leads to a restart of the algorithm (setting beta = 0) after two
        // consecutive search directions are not orthogonal anymore. In other words, if this condition
        // is met:
        //
        // `|\nabla f_k^T * \nabla f_{k-1}| / | \nabla f_k ||^2 >= v`
        //
        // A typical value for `v` is 0.1.
        .restart_orthogonality(0.1);

    // Run solver
    let res = Executor::new(operator, solver)
        .configure(|state| state.param(init_param).max_iters(20).target_cost(0.0))
        .add_observer(SlogLogger::term(), ObserverMode::Always)
        .run()?;

    // Wait a second (lets the logger flush everything before printing to screen again)
    std::thread::sleep(std::time::Duration::from_secs(1));

    // Print result
    println!("{}", res);
    Ok(())
}

fn main() {
    if let Err(ref e) = run() {
        println!("{}", e);
    }
}
