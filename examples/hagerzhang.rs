// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

extern crate argmin;
use argmin::prelude::*;
use argmin::solver::linesearch::HagerZhangLineSearch;

struct problem {}

// f(x)=(x−3) x^{3} (x−6)^{4}
pub fn cf(params: &Vec<f64>) -> f64 {
    let x = params[0];
    let value: f64 = (x - 3.0) * x.powi(3) * (x - 6.0).powi(4);
    println!("x: {}, value: {}", x, &value);
    value
}

// df(x) = (x - 6)**5 *
// (x**3*(x - 6)*(x - 3)**(x**4)*(x + 4*(x - 3)*log(x - 3)) + 6*(x - 3)**(x**4 + 1)) /
// (x - 3)
pub fn cf_deriv(params: &Vec<f64>) -> Vec<f64> {
    let mut value: Vec<f64> = std::vec::Vec::new();
    for x in params {
        let var1 = x.powi(3) * (x - 6.0).powi(4);
        let var2 = 4.0 * x.powi(3) * (x - 6.0).powi(3) * (x - 3.0);
        let var3 = 3.0 * x.powi(2) * (x - 6.0).powi(4) * (x - 3.0);
        value.push(var1 + var2 + var3);
        println!("x: {}, deriv: {}", x, var1 + var2 + var3);
    }
    value
}

impl ArgminOp for problem {
    type Param = Vec<f64>;
    type Output = f64;
    type Hessian = ();
    type Jacobian = ();
    type Float = f64;

    fn apply(&self, param: &Vec<f64>) -> Result<f64, Error> {
        Ok(cf(param))
    }

    fn gradient(&self, param: &Vec<f64>) -> Result<Vec<f64>, Error> {
        Ok(cf_deriv(param))
    }
}

fn run() -> Result<(), Error> {
    // Define inital parameter vector
    let init_param: Vec<f64> = vec![4.0];

    // Problem definition
    let operator = problem {};

    // Set up line search method
    let mut solver = HagerZhangLineSearch::new();

    // The following parameters do not follow the builder pattern because they are part of the
    // ArgminLineSearch trait which needs to be object safe.

    // Set search direction
    solver.set_search_direction(vec![1.0]);

    // Set initial step length
    solver.set_init_alpha(1.5)?;

    let init_cost = operator.apply(&init_param)?;
    let init_grad = operator.gradient(&init_param)?;

    println!("init_cost: {:?}", init_cost);
    println!("init_grad: {:?}", init_grad);

    // Run solver
    let res = Executor::new(operator, solver, init_param)
        .add_observer(ArgminSlogLogger::term(), ObserverMode::Always)
        .max_iters(10)
        // the following two are optional. If they are not provided, they will be computed
        .cost(init_cost)
        .grad(init_grad)
        .run()?;

    // Wait a second (lets the logger flush everything before printing again)
    std::thread::sleep(std::time::Duration::from_secs(1));

    // Print Result
    println!("results: {}", res);
    Ok(())
}

fn main() {
    if let Err(ref e) = run() {
        println!("{}", e);
    }
}
