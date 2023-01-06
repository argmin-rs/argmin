// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use argmin::core::observers::{ObserverMode, SlogLogger};
use argmin::core::{Error, Executor, Operator};
use argmin::solver::conjugategradient::ConjugateGradient;

struct MyProblem {}

impl Operator for MyProblem {
    type Param = Vec<f64>;
    type Output = Vec<f64>;

    fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        Ok(vec![4.0 * p[0] + 1.0 * p[1], 1.0 * p[0] + 3.0 * p[1]])
    }
}

fn run() -> Result<(), Error> {
    // Define initial parameter vector
    let init_param: Vec<f64> = vec![2.0, 1.0];

    // Define the right hand side `b` of `A * x = b`
    let b = vec![1.0, 2.0];

    // Set up operator
    let operator = MyProblem {};

    // Set up the solver
    let solver: ConjugateGradient<_, f64> = ConjugateGradient::new(b);

    // Run solver
    let res = Executor::new(operator, solver)
        .configure(|state| state.param(init_param).max_iters(2))
        .add_observer(SlogLogger::term(), ObserverMode::Always)
        .run()?;

    // Wait a second (lets the logger flush everything before printing to screen again)
    std::thread::sleep(std::time::Duration::from_secs(1));

    // Print result
    println!("{res}");
    Ok(())
}

fn main() {
    if let Err(ref e) = run() {
        println!("{e}");
    }
}
