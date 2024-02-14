// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use argmin::{
    core::{observers::ObserverMode, CostFunction, Error, Executor},
    solver::neldermead::NelderMead,
};
use argmin_observer_slog::SlogLogger;
use argmin_testfunctions::rosenbrock;
use ndarray::{array, Array1};

struct Rosenbrock {}

impl CostFunction for Rosenbrock {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        Ok(rosenbrock(&p.to_vec()))
    }
}

fn run() -> Result<(), Error> {
    // Define cost function
    let cost = Rosenbrock {};

    // Set up solver -- note that the proper choice of the vertices is very important!
    let solver = NelderMead::new(vec![
        // array![-2.0, 3.0],
        // array![-2.0, -1.0],
        // array![2.0, -1.0],
        array![-1.0, 3.0],
        array![2.0, 1.5],
        array![2.0, -1.0],
    ])
    .with_sd_tolerance(0.0001)?;

    // Run solver
    let res = Executor::new(cost, solver)
        .configure(|state| state.max_iters(100))
        .add_observer(SlogLogger::term(), ObserverMode::Always)
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
