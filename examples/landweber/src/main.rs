// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use argmin::{
    core::{observers::ObserverMode, Error, Executor, Gradient},
    solver::landweber::Landweber,
};
use argmin_observer_slog::SlogLogger;
use argmin_testfunctions::rosenbrock_derivative;

struct Rosenbrock {}

impl Gradient for Rosenbrock {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
        Ok(rosenbrock_derivative(p))
    }
}

fn run() -> Result<(), Error> {
    // define initial parameter vector
    let init_param: Vec<f64> = vec![1.2, 1.2];
    let operator = Rosenbrock {};

    let iters = 10;
    let solver = Landweber::new(0.001);

    let res = Executor::new(operator, solver)
        .configure(|state| state.param(init_param).max_iters(iters))
        .add_observer(SlogLogger::term(), ObserverMode::Always)
        .run()?;

    println!("{res}");
    Ok(())
}

fn main() {
    if let Err(ref e) = run() {
        println!("{e}");
    }
}
