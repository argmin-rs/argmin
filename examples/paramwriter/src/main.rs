// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use argmin::{
    core::{observers::ObserverMode, CostFunction, Error, Executor, Gradient},
    solver::{gradientdescent::SteepestDescent, linesearch::MoreThuenteLineSearch},
};
use argmin_observer_paramwriter::{ParamWriter, ParamWriterFormat};
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
    // Define cost function
    let cost = Rosenbrock {};

    // Define initial parameter vector
    let init_param: Vec<f64> = vec![1.2, 1.2];

    // Pick a line search.
    let linesearch = MoreThuenteLineSearch::new();

    // Set up solver
    let solver = SteepestDescent::new(linesearch);

    // Create writer
    // Set serializer to Binary (will use the `bincode` crate for serializing)
    let writer = ParamWriter::new("params", "param", ParamWriterFormat::Binary);

    // Create writer which only saves new best ones
    // Set serializer to JSON
    let writer2 = ParamWriter::new("params", "best", ParamWriterFormat::JSON);

    let res = Executor::new(cost, solver)
        .configure(|state| state.param(init_param).max_iters(10))
        .add_observer(writer, ObserverMode::Every(3))
        .add_observer(writer2, ObserverMode::NewBest)
        .run()?;

    // Print result
    println!("{res}");
    Ok(())
}

fn main() {
    if let Err(ref e) = run() {
        println!("{e}");
        std::process::exit(1);
    }
}
