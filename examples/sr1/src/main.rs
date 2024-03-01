// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

#[allow(unused_imports)]
use argmin::{
    core::{observers::ObserverMode, CostFunction, Error, Executor, Gradient},
    solver::{
        linesearch::{HagerZhangLineSearch, MoreThuenteLineSearch},
        quasinewton::SR1,
    },
};
use argmin_observer_slog::SlogLogger;
use argmin_testfunctions::{styblinski_tang, styblinski_tang_derivative};
use ndarray::{array, Array1, Array2};

struct StyblinskiTang {}

impl CostFunction for StyblinskiTang {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        Ok(styblinski_tang(&p.to_vec()))
    }
}
impl Gradient for StyblinskiTang {
    type Param = Array1<f64>;
    type Gradient = Array1<f64>;

    fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
        Ok(styblinski_tang_derivative(&p.to_vec()).into())
    }
}

fn run() -> Result<(), Error> {
    // Define cost function
    let cost = StyblinskiTang {};

    // Define initial parameter vector
    // let init_param: Array1<f64> = array![-1.2, 1.0, -5.0, 2.0, 3.0, 2.0, 4.0, 5.0];
    let init_param: Array1<f64> = array![5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0];
    let init_hessian: Array2<f64> = Array2::eye(8);

    // set up a line search
    let linesearch = MoreThuenteLineSearch::new().with_c(1e-4, 0.9)?;
    // let linesearch = HagerZhangLineSearch::new();

    // Set up solver
    let solver = SR1::new(linesearch);

    // Run solver
    let res = Executor::new(cost, solver)
        .configure(|state| {
            state
                .param(init_param)
                .inv_hessian(init_hessian)
                .max_iters(1000)
        })
        .add_observer(SlogLogger::term(), ObserverMode::Always)
        .run()?;

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
