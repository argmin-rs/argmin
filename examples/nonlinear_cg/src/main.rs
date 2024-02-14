// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use argmin::{
    core::{observers::ObserverMode, CostFunction, Error, Executor, Gradient},
    solver::{
        conjugategradient::{beta::PolakRibiere, NonlinearConjugateGradient},
        linesearch::MoreThuenteLineSearch,
    },
};
use argmin_observer_slog::SlogLogger;
use argmin_testfunctions::{rosenbrock, rosenbrock_derivative};

struct Rosenbrock {}

impl CostFunction for Rosenbrock {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        Ok(rosenbrock(p))
    }
}

impl Gradient for Rosenbrock {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
        Ok(rosenbrock_derivative(p))
    }
}

fn run() -> Result<(), Error> {
    // Set up cost function
    let operator = Rosenbrock {};

    // define initial parameter vector
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

    // Print result
    println!("{res}");
    Ok(())
}

fn main() {
    if let Err(ref e) = run() {
        println!("{e}");
    }
}
