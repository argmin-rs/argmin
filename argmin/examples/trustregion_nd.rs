// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use argmin::core::{
    ArgminSlogLogger, CostFunction, Error, Executor, Gradient, Hessian, ObserverMode,
};
#[allow(unused_imports)]
use argmin::solver::trustregion::{CauchyPoint, Dogleg, Steihaug, TrustRegion};
use argmin_testfunctions::{rosenbrock_2d, rosenbrock_2d_derivative, rosenbrock_2d_hessian};
use ndarray::{Array, Array1, Array2};

struct Rosenbrock {
    a: f64,
    b: f64,
}

impl CostFunction for Rosenbrock {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        Ok(rosenbrock_2d(&p.to_vec(), self.a, self.b))
    }
}
impl Gradient for Rosenbrock {
    type Param = Array1<f64>;
    type Gradient = Array1<f64>;

    fn gradient(&self, p: &Self::Param) -> Result<Self::Param, Error> {
        Ok(Array1::from(rosenbrock_2d_derivative(
            &p.to_vec(),
            self.a,
            self.b,
        )))
    }
}

impl Hessian for Rosenbrock {
    type Param = Array1<f64>;
    type Hessian = Array2<f64>;

    fn hessian(&self, p: &Self::Param) -> Result<Self::Hessian, Error> {
        let h = rosenbrock_2d_hessian(&p.to_vec(), self.a, self.b);
        Ok(Array::from_shape_vec((2, 2), h)?)
    }
}

fn run() -> Result<(), Error> {
    // Define cost function
    let cost = Rosenbrock { a: 1.0, b: 100.0 };

    // Define inital parameter vector
    // easy case
    // let init_param: Array1<f64> = Array1::from_vec(vec![1.2, 1.2]);
    // tough case
    let init_param: Array1<f64> = Array1::from(vec![-1.2, 1.0]);

    // Set up the subproblem
    // let subproblem = Steihaug::new().max_iters(2);
    // let subproblem = CauchyPoint::new();
    let subproblem = Dogleg::new();

    // Set up solver
    let solver = TrustRegion::new(subproblem);

    // Run solver
    let res = Executor::new(cost, solver)
        .configure(|config| config.param(init_param).max_iters(50))
        .add_observer(ArgminSlogLogger::term(), ObserverMode::Always)
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
