// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use argmin::{
    core::{observers::ObserverMode, CostFunction, Error, Executor, Gradient},
    solver::{linesearch::MoreThuenteLineSearch, quasinewton::BFGS},
};
use argmin_observer_slog::SlogLogger;
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

    fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
        Ok((*p).forward_diff(&|x| rosenbrock(&x.to_vec(), self.a, self.b)))
    }
}

fn run() -> Result<(), Error> {
    // Define cost function
    let cost = Rosenbrock { a: 1.0, b: 100.0 };

    // Define initial parameter vector
    // let init_param: Array1<f64> = array![-1.2, 1.0];
    // let init_hessian: Array2<f64> = Array2::eye(2);
    let init_param: Array1<f64> = array![-1.2, 1.0, -10.0, 2.0, 3.0, 2.0, 4.0, 10.0];
    let init_hessian: Array2<f64> = Array2::eye(8);

    // set up a line search
    let linesearch = MoreThuenteLineSearch::new().with_c(1e-4, 0.9)?;

    // Set up solver
    let solver = BFGS::new(linesearch);

    // Run solver
    let res = Executor::new(cost, solver)
        .configure(|state| {
            state
                .param(init_param)
                .inv_hessian(init_hessian)
                .max_iters(60)
        })
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
