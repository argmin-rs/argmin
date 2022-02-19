// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use argmin::core::{ArgminOp, ArgminSlogLogger, CheckpointMode, Error, Executor, ObserverMode};
use argmin::solver::landweber::Landweber;
use argmin_testfunctions::{rosenbrock_2d, rosenbrock_2d_derivative};

#[derive(Default)]
struct Rosenbrock {}

impl ArgminOp for Rosenbrock {
    type Param = Vec<f64>;
    type Output = f64;
    type Hessian = ();
    type Jacobian = ();
    type Float = f64;

    fn apply(&self, p: &Vec<f64>) -> Result<f64, Error> {
        Ok(rosenbrock_2d(p, 1.0, 100.0))
    }

    fn gradient(&self, p: &Vec<f64>) -> Result<Vec<f64>, Error> {
        Ok(rosenbrock_2d_derivative(p, 1.0, 100.0))
    }
}

fn run() -> Result<(), Error> {
    // define inital parameter vector
    let init_param: Vec<f64> = vec![1.2, 1.2];

    let iters = 35;
    let solver = Landweber::new(0.001);

    let res = Executor::from_checkpoint(".checkpoints/landweber_exec.arg", Rosenbrock {})
        .unwrap_or_else(|_| {
            Executor::new(Rosenbrock {}, solver).configure(|config| config.param(init_param))
        })
        .max_iters(iters)
        .checkpoint_dir(".checkpoints")
        .checkpoint_name("landweber_exec")
        .checkpoint_mode(CheckpointMode::Every(20))
        .add_observer(ArgminSlogLogger::term(), ObserverMode::Always)
        .run()?;

    // Wait a second (lets the logger flush everything before printing to screen again)
    std::thread::sleep(std::time::Duration::from_secs(1));
    println!("{}", res);
    Ok(())
}

fn main() {
    if let Err(ref e) = run() {
        println!("{}", e);
    }
}
