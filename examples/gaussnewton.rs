// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

extern crate argmin;
extern crate ndarray;
use argmin::prelude::*;
use argmin::solver::gaussnewton::GaussNewton;
// use argmin::solver::linesearch::MoreThuenteLineSearch;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

type Rate = f64;
type S = f64;
type Measurement = (S, Rate);

// Example taken from Wikipedia: https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm
// Model used in this example:
// `rate = (V_{max} * [S]) / (K_M + [S]) `
// where `V_{max}` and `K_M` are the sought parameters and `[S]` and `rate` is the measured data.
#[derive(Clone, Default, Serialize, Deserialize)]
struct Problem {
    data: Vec<Measurement>,
}

impl ArgminOp for Problem {
    type Param = Array1<f64>;
    type Output = Array1<f64>;
    type Hessian = ();
    type Jacobian = Array2<f64>;

    fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        Ok(self
            .data
            .iter()
            .map(|(s, rate)| rate - (p[0] * s) / (p[1] + s))
            .collect::<Array1<f64>>())
    }

    fn jacobian(&self, p: &Self::Param) -> Result<Self::Jacobian, Error> {
        Ok(Array2::from_shape_fn((7, 2), |(si, i)| {
            if i == 0 {
                -self.data[si].0 / (p[1] + self.data[si].0)
            } else {
                p[0] * self.data[si].0 / (p[1] + self.data[si].0).powi(2)
            }
        }))
    }
}

fn run() -> Result<(), Error> {
    // Define cost function
    // Example taken from Wikipedia: https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm
    let cost = Problem {
        data: vec![
            (0.038, 0.050),
            (0.194, 0.127),
            (0.425, 0.094),
            (0.626, 0.2122),
            (1.253, 0.2729),
            (2.5, 0.2665),
            (3.74, 0.3317),
        ],
    };

    // let linesearch = MoreThuenteLineSearch::new();

    // Define initial parameter vector
    let init_param: Array1<f64> = Array1::from(vec![0.9, 0.2]);

    // Set up solver
    let solver: GaussNewton = GaussNewton::new();

    // Run solver
    let res = Executor::new(cost, solver, init_param)
        .add_observer(ArgminSlogLogger::term(), ObserverMode::Always)
        .max_iters(10)
        .run()?;

    // Wait a second (lets the logger flush everything before printing again)
    std::thread::sleep(std::time::Duration::from_secs(1));

    // Print result
    println!("{}", res);
    Ok(())
}

fn main() {
    if let Err(ref e) = run() {
        println!("{} {}", e.as_fail(), e.backtrace());
        std::process::exit(1);
    }
}
