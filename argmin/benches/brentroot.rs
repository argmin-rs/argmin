// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use criterion::{criterion_group, criterion_main, Criterion};

use argmin::core::{CostFunction, Error, Executor};
use argmin::solver::brent::BrentRoot;

/// Test function generalise from Wikipedia example
struct TestFunc {
    zero1: f64,
    zero2: f64,
}

impl CostFunction for TestFunc {
    // one dimensional problem, no vector needed
    type Param = f64;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        Ok((p + self.zero1) * (p - self.zero2) * (p - self.zero2))
    }
}

fn run() -> Result<(), Error> {
    let cost = TestFunc {
        zero1: 3.,
        zero2: -1.,
    };
    let init_param = 0.5;
    let solver = BrentRoot::new(-4., 0.5, 1e-11);

    let _res = Executor::new(cost, solver)
        .configure(|state| state.param(init_param).max_iters(100))
        // .add_observer(SlogLogger::term(), ObserverMode::Always)
        .run()
        .unwrap();

    Ok(())
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("BrentRoot", |b| {
        b.iter(|| run().expect("Benchmark should run without errors"))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
