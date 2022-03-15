// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use argmin::core::{CostFunction, Error, Executor, Gradient, Hessian, ObserverMode, SlogLogger};
use argmin::solver::quasinewton::SR1TrustRegion;
#[allow(unused_imports)]
use argmin::solver::trustregion::{CauchyPoint, Dogleg, Steihaug, TrustRegion};
use argmin_testfunctions::rosenbrock;
use finitediff::FiniteDiff;
use ndarray::{array, Array1, Array2};

struct Rosenbrock {
    a: f64,
    b: f64,
}

impl CostFunction for Rosenbrock {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        Ok(rosenbrock(&p.to_vec(), self.a, self.b))
    }
}
impl Gradient for Rosenbrock {
    type Param = Array1<f64>;
    type Gradient = Array1<f64>;

    fn gradient(&self, p: &Self::Param) -> Result<Self::Param, Error> {
        Ok((*p).forward_diff(&|x| rosenbrock(&x.to_vec(), self.a, self.b)))
    }
}

impl Hessian for Rosenbrock {
    type Param = Array1<f64>;
    type Hessian = Array2<f64>;

    fn hessian(&self, p: &Self::Param) -> Result<Self::Hessian, Error> {
        Ok((*p).forward_hessian(&|x| self.gradient(x).unwrap()))
    }
}

fn run() -> Result<(), Error> {
    // Define cost function
    let cost = Rosenbrock { a: 1.0, b: 100.0 };

    // Define initial parameter vector
    let init_param: Array1<f64> = array![-1.2, 1.0];
    // let init_param: Array1<f64> = array![1.2, 1.0];
    let init_hessian: Array2<f64> = Array2::eye(2);
    // let init_param: Array1<f64> = array![-1.2, 1.0, -10.0, 2.0, 3.0, 2.0, 4.0, 10.0];
    // let init_hessian: Array2<f64> = Array2::eye(8);

    // Set up the subproblem
    let subproblem = Steihaug::new().max_iters(20);
    // let subproblem = CauchyPoint::new();
    // let subproblem = Dogleg::new();

    // Set up solver
    let solver = SR1TrustRegion::new(subproblem);

    // Run solver
    let res = Executor::new(cost, solver)
        .configure(|state| {
            state
                .param(init_param)
                .hessian(init_hessian)
                .max_iters(1000)
        })
        .add_observer(SlogLogger::term(), ObserverMode::Always)
        .run()?;

    // Wait a second (lets the observer flush everything before printing again)
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
