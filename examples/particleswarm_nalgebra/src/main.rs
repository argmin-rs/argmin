// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use argmin::core::{CostFunction, Error, Executor};
use argmin::solver::particleswarm::ParticleSwarm;
use argmin_testfunctions::himmelblau;
use nalgebra::{dvector, DVector};

struct Himmelblau {}

impl CostFunction for Himmelblau {
    type Param = DVector<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        Ok(himmelblau(param.data.as_vec()))
    }
}

fn run() -> Result<(), Error> {
    let cost_function = Himmelblau {};

    let solver = ParticleSwarm::new((dvector![-4.0, -4.0], dvector![4.0, 4.0]), 40);

    let res = Executor::new(cost_function, solver)
        .configure(|state| state.max_iters(100))
        .run()?;

    // Print Result
    println!("{res}");

    Ok(())
}

fn main() {
    if let Err(ref e) = run() {
        println!("{e}");
    }
}
