// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

extern crate argmin;
extern crate ndarray;
use argmin::prelude::*;
use argmin::solver::quasinewton::BFGS;
use argmin_core::finitediff::*;
// use argmin::testfunctions::{rosenbrock_2d, rosenbrock_2d_derivative};
use argmin::testfunctions::rosenbrock;
use ndarray::{array, Array1, Array2};

#[derive(Clone)]
struct Rosenbrock {
    a: f64,
    b: f64,
}

impl ArgminOperator for Rosenbrock {
    type Parameters = Array1<f64>;
    type OperatorOutput = f64;
    type Hessian = Array2<f64>;

    fn apply(&self, p: &Self::Parameters) -> Result<Self::OperatorOutput, Error> {
        Ok(rosenbrock(&p.to_vec(), self.a, self.b))
    }

    fn gradient(&self, p: &Self::Parameters) -> Result<Self::Parameters, Error> {
        Ok((*p).forward_diff(&|x| rosenbrock(&x.to_vec(), self.a, self.b)))
    }
}

fn run() -> Result<(), Error> {
    // Define cost function
    let cost = Rosenbrock { a: 1.0, b: 100.0 };

    // Define initial parameter vector
    // let init_param: Array1<f64> = array![-1.2, 1.0];
    let init_param: Array1<f64> = array![-1.2, 1.0, -10.0, 2.0, 3.0, 2.0, 4.0, 10.0];
    let init_hessian: Array2<f64> = Array2::eye(8);

    // Set up solver
    let mut solver = BFGS::new(&cost, init_param, init_hessian);

    // Set maximum number of iterations
    solver.set_max_iters(800);

    // Attach a logger
    solver.add_logger(ArgminSlogLogger::term());

    // Run solver
    solver.run()?;

    // Wait a second (lets the logger flush everything before printing again)
    std::thread::sleep(std::time::Duration::from_secs(1));

    // Print result
    println!("{}", solver.result());
    Ok(())
}

fn main() {
    if let Err(ref e) = run() {
        println!("{} {}", e.as_fail(), e.backtrace());
        std::process::exit(1);
    }
}
