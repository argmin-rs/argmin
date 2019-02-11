// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

#![allow(unused_imports)]

extern crate argmin;
use argmin::prelude::*;
use argmin::solver::gradientdescent::SteepestDescent;
use argmin::solver::linesearch::BacktrackingLineSearch;
use argmin::solver::linesearch::HagerZhangLineSearch;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::testfunctions::{rosenbrock_2d, rosenbrock_2d_derivative};
use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Serialize, Deserialize)]
struct Rosenbrock {
    a: f64,
    b: f64,
}

impl ArgminOp for Rosenbrock {
    type Param = Vec<f64>;
    type Output = f64;
    type Hessian = ();

    fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        Ok(rosenbrock_2d(p, self.a, self.b))
    }

    fn gradient(&self, p: &Self::Param) -> Result<Self::Param, Error> {
        Ok(rosenbrock_2d_derivative(p, self.a, self.b))
    }
}

fn run() -> Result<(), Error> {
    // Define cost function (must implement `ArgminOperator`)
    let cost = Rosenbrock { a: 1.0, b: 100.0 };

    // Define initial parameter vector
    // easy case
    let init_param: Vec<f64> = vec![1.2, 1.2];
    // tough case
    // let init_param: Vec<f64> = vec![-1.2, 1.0];

    // Pick a line search.
    // let linesearch = HagerZhangLineSearch::new(cost.clone());
    let linesearch = MoreThuenteLineSearch::new(cost.clone());
    // let linesearch = BacktrackingLineSearch::new(cost.clone());

    // Set up solver
    let mut solver = SteepestDescent::new(cost, init_param, linesearch)?;

    // Set maximum number of iterations
    solver.set_max_iters(10_000);

    // Attach a logger which will output information in each iteration.
    solver.add_logger(ArgminSlogLogger::term_noblock());

    // Run the solver
    solver.run()?;

    // Wait a second (lets the logger flush everything first)
    std::thread::sleep(std::time::Duration::from_secs(1));

    // print result
    println!("{}", solver.result());
    Ok(())
}

fn main() {
    if let Err(ref e) = run() {
        println!("{} {}", e.as_fail(), e.backtrace());
        std::process::exit(1);
    }
}

// DUMP (ignore)
// #[derive(Clone, ArgminOperator)]
// #[output_type(f64)]
// #[parameters_type(Vec<f64>)]
// #[hessian_type(())]
// #[cost_function(rosenbrock)]
// #[gradient(rosenbrock_gradient)]
// struct MyProblem {}
