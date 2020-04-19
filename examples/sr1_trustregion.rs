// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

extern crate argmin;
extern crate argmin_testfunctions;
extern crate finitediff;
extern crate ndarray;
use argmin::prelude::*;
use argmin::solver::quasinewton::SR1TrustRegion;
#[allow(unused_imports)]
use argmin::solver::trustregion::{CauchyPoint, Dogleg, Steihaug, TrustRegion};
use argmin_testfunctions::rosenbrock;
use finitediff::*;
use ndarray::{array, Array1, Array2};

struct Rosenbrock {
    a: f64,
    b: f64,
}

impl ArgminOp for Rosenbrock {
    type Param = Array1<f64>;
    type Output = f64;
    type Hessian = Array2<f64>;
    type Jacobian = ();
    type Float = f64;

    fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        Ok(rosenbrock(&p.to_vec(), self.a, self.b))
    }

    fn gradient(&self, p: &Self::Param) -> Result<Self::Param, Error> {
        Ok((*p).forward_diff(&|x| rosenbrock(&x.to_vec(), self.a, self.b)))
        // Ok(ndarray::Array1::from_vec(rosenbrock_2d_derivative(
        //     &p.to_vec(),
        //     self.a,
        //     self.b,
        // )))
    }

    fn hessian(&self, p: &Self::Param) -> Result<Self::Hessian, Error> {
        Ok((*p).forward_hessian(&|x| self.gradient(&x).unwrap()))
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
    let res = Executor::new(cost, solver, init_param)
        .add_observer(ArgminSlogLogger::term(), ObserverMode::Always)
        .max_iters(1000)
        .hessian(init_hessian)
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
