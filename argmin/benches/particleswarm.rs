// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.
use criterion::{criterion_group, criterion_main, Criterion};

use argmin::core::{CostFunction, Error, Executor};
use argmin::solver::particleswarm::ParticleSwarm;
use argmin_testfunctions::himmelblau;

struct Himmelblau {}

impl CostFunction for Himmelblau {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        Ok(himmelblau(param))
    }
}

fn run() -> Result<(), Error> {
    let cost_function = Himmelblau {};

    let solver = ParticleSwarm::new((vec![-4.0, -4.0], vec![4.0, 4.0]), 40);

    let res = Executor::new(cost_function, solver)
        .configure(|state| state.max_iters(100))
        .run()?;
    Ok(())
}


fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("ParticleSwarm", |b| b.iter(|| run()));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

