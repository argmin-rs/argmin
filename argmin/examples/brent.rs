// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use argmin::core::{ArgminOp, ArgminSlogLogger, Error, Executor, ObserverMode};
use argmin::solver::brent::Brent;

/// Test function generalise from Wikipedia example
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
    type Float = f64;

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

    let res = Executor::new(cost, solver)
        .configure(|config| config.param(init_param).max_iters(100))
        .add_observer(ArgminSlogLogger::term(), ObserverMode::Always)
        .run()
        .unwrap();
    println!("Result of brent:\n{}", res);
}
