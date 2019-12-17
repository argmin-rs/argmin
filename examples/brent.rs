// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

extern crate argmin;
use argmin::prelude::*;
use argmin::solver::brent::Brent;
use serde::{Deserialize, Serialize};

/// Test function generalise from Wikipedia example
#[derive(Clone, Default, Serialize, Deserialize)]
struct TestFunc {
    zero1: f64,
    zero2: f64,
}

impl ArgminOp for TestFunc {
    // one dimensional problem, no vector needed
    type Param = f64;
    type Output = f64;
    type Hessian = ();
    type Jacobian = ();

    fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        Ok((p + self.zero1) * (p - self.zero2) * (p - self.zero2))
    }
}

fn main() {
    let cost = TestFunc {
        zero1: 3.,
        zero2: -1.,
    };
    let init_param = 0.5;
    let solver = Brent::new(-4., 0.5, 1e-11);

    let res = Executor::new(cost, solver, init_param)
        .add_observer(ArgminSlogLogger::term(), ObserverMode::Always)
        .max_iters(100)
        .run()
        .unwrap();
    println!("Result of brent:\n{}", res);
}
