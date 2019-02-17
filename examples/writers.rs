// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

extern crate argmin;
extern crate ndarray;
use argmin::prelude::*;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::BFGS;
use argmin::testfunctions::rosenbrock;
use argmin_core::finitediff::*;
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

    fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        Ok(rosenbrock(&p.to_vec(), self.a, self.b))
    }

    fn gradient(&self, p: &Self::Param) -> Result<Self::Param, Error> {
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

    // set up a line search
    let linesearch = MoreThuenteLineSearch::new(cost.clone());

    // Set up solver
    let mut solver = BFGS::new(cost, init_param, init_hessian, linesearch);

    // Set maximum number of iterations
    solver.set_max_iters(10);

    // Attach a logger
    solver.add_logger(ArgminSlogLogger::term());

    // Attach a writer
    solver.add_writer(WriteToFile::new("params"));

    // Set how often parameter should be saved
    solver.set_writer_mode(WriterMode::Every(3));

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
