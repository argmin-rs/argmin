// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use argmin::prelude::*;
use argmin::solver::conjugategradient::NonlinearConjugateGradient;
use argmin::solver::conjugategradient::PolakRibiere;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin_testfunctions::{rosenbrock_2d, rosenbrock_2d_derivative};

struct Rosenbrock {}

impl ArgminOp for Rosenbrock {
    type Param = Vec<f64>;
    type Output = f64;
    type Hessian = ();
    type Jacobian = ();
    type Float = f64;

    fn apply(&self, p: &Vec<f64>) -> Result<f64, Error> {
        Ok(rosenbrock_2d(p, 1.0, 100.0))
    }

    fn gradient(&self, p: &Vec<f64>) -> Result<Vec<f64>, Error> {
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
    let solver = NonlinearConjugateGradient::new(linesearch, beta_method)?
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
    let res = Executor::new(operator, solver, init_param)
        .add_observer(ArgminSlogLogger::term(), ObserverMode::Always)
        .max_iters(20)
        // Set target cost function value
        .target_cost(0.0)
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
