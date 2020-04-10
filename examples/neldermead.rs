// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

extern crate argmin;
extern crate argmin_testfunctions;
extern crate ndarray;
use argmin::prelude::*;
use argmin::solver::neldermead::NelderMead;
use argmin_testfunctions::rosenbrock;
use ndarray::{array, Array1, Array2};
use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Serialize, Deserialize)]
struct Rosenbrock {
    a: f64,
    b: f64,
}

impl ArgminOp for Rosenbrock {
    type Param = Array1<f64>;
    type Output = f64;
    type Hessian = Array2<f64>;
    type Jacobian = ();

    fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        Ok(rosenbrock(&p.to_vec(), self.a, self.b))
    }
}

fn run() -> Result<(), Error> {
    // Define cost function
    let cost = Rosenbrock { a: 1.0, b: 100.0 };

    // Set up solver -- note that the proper choice of the vertices is very important!
    let solver = NelderMead::new()
        .with_initial_params(vec![
            // array![-2.0, 3.0],
            // array![-2.0, -1.0],
            // array![2.0, -1.0],
            array![-1.0, 3.0],
            array![2.0, 1.5],
            array![2.0, -1.0],
        ])
        .sd_tolerance(0.0001);

    // Run solver
    let res = Executor::new(cost, solver, array![])
        .add_observer(ArgminSlogLogger::term(), ObserverMode::Always)
        .max_iters(100)
        .run()?;

    // Wait a second (lets the logger flush everything before printing again)
    std::thread::sleep(std::time::Duration::from_secs(1));

    // Print result
    println!("{}", res);
    Ok(())
}

fn main() {
    if let Err(ref e) = run() {
        println!("{}", e);
        std::process::exit(1);
    }
}
