// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

// use argmin::core::{ArgminOp, ArgminSlogLogger, Error, Executor, ObserverMode};
use argmin::core::*;
use argmin::solver::simplex::Simplex;

struct Problem {
    c: Vec<f64>,
    b: Vec<f64>,
    a: Vec<Vec<f64>>,
}

impl LinearProgram for Problem {
    type Param = Vec<Self::Float>;
    type Float = f64;

    fn c(&self) -> Result<&[Self::Float], Error> {
        Ok(&self.c)
    }

    fn b(&self) -> Result<&[Self::Float], Error> {
        Ok(&self.b)
    }

    fn A(&self) -> Result<&[Vec<Self::Float>], Error> {
        Ok(&self.a)
    }
}

fn run() -> Result<(), Error> {
    // let problem = Problem {
    //     c: [-5.0f64, -6.0, -6.0].to_vec(),
    //     b: [10.0f64, 10.0, 10.0].to_vec(),
    //     a: [
    //         [1.0, 2.0, 2.0].to_vec(),
    //         [2.0, 1.0, 2.0].to_vec(),
    //         [2.0, 2.0, 1.0].to_vec(),
    //     ]
    //     .to_vec(),
    // };
    // let problem = Problem {
    //     c: [-3.0f64, -1.0].to_vec(),
    //     b: [1.0f64, 1.0, 2.0].to_vec(),
    //     a: [
    //         [-1.0, 1.0].to_vec(),
    //         [1.0, -1.0].to_vec(),
    //         [0.0, 1.0].to_vec(),
    //     ]
    //     .to_vec(),
    // };
    let problem = Problem {
        c: [-12.0f64, -8.0].to_vec(),
        b: [80.0f64, 100.0, 75.0].to_vec(),
        a: [
            [4.0, 2.0].to_vec(),
            [2.0, 3.0].to_vec(),
            [5.0, 1.0].to_vec(),
        ]
        .to_vec(),
    };

    let solver: Simplex<f64> = Simplex::new();

    // let init_param  = [1.0, 2.0, 3.0].to_vec();
    let init_param = vec![];
    let res = Executor::new(problem, solver, init_param)
        // .add_observer(ArgminSlogLogger::term(), ObserverMode::Always)
        .max_iters(3)
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
