// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use argmin::core::observers::{ObserverMode, SlogLogger};
use argmin::core::{Error, Executor, Jacobian, Operator};
use argmin::solver::gaussnewton::GaussNewton;

use nalgebra::{DMatrix, DVector};

type Rate = f64;
type S = f64;
type Measurement = (S, Rate);

// Example taken from Wikipedia: https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm
// Model used in this example:
// `rate = (V_{max} * [S]) / (K_M + [S]) `
// where `V_{max}` and `K_M` are the sought parameters and `[S]` and `rate` is the measured data.
struct Problem {
    data: Vec<Measurement>,
}

impl Operator for Problem {
    type Param = DVector<f64>;
    type Output = DVector<f64>;

    fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        Ok(DVector::from_vec(
            self.data
                .iter()
                .map(|(s, rate)| rate - (p[0] * s) / (p[1] + s))
                .collect(),
        ))
    }
}

impl Jacobian for Problem {
    type Param = DVector<f64>;
    type Jacobian = DMatrix<f64>;

    fn jacobian(&self, p: &Self::Param) -> Result<Self::Jacobian, Error> {
        Ok(DMatrix::from_fn(7, 2, |si, i| {
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

    // Define initial parameter vector
    let init_param: DVector<f64> = DVector::from_vec(vec![0.9, 0.2]);

    // Set up solver
    let solver: GaussNewton<f64> = GaussNewton::new();

    // Run solver
    let res = Executor::new(cost, solver)
        .configure(|state| state.param(init_param).max_iters(10))
        .add_observer(SlogLogger::term(), ObserverMode::Always)
        .run()?;

    // Wait a second (lets the logger flush everything before printing again)
    std::thread::sleep(std::time::Duration::from_secs(1));

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
