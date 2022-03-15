// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use argmin::core::{CostFunction, Error, Executor, ObserverMode, SlogLogger};
use argmin::solver::goldensectionsearch::GoldenSectionSearch;

/// Test function from Wikipedia example
struct TestFunc {}

impl CostFunction for TestFunc {
    // one dimensional problem, no vector needed
    type Param = f32;
    type Output = f32;

    fn cost(&self, x: &Self::Param) -> Result<Self::Output, Error> {
        // In interval [2.5, 2.5]
        // Min at 1.0
        // Max at -1.666 (multiply by -1.0 to test)
        Ok((x + 3.0) * (x - 1.0).powi(2))
    }
}

fn main() {
    let cost = TestFunc {};
    let init_param = -0.5;
    let solver = GoldenSectionSearch::new(-2.5, 3.0).tolerance(0.0001);

    let res = Executor::new(cost, solver)
        .configure(|state| state.param(init_param).max_iters(100))
        .add_observer(SlogLogger::term(), ObserverMode::Always)
        .run()
        .unwrap();
    println!("Result of golden section search:\n{}", res);
}
