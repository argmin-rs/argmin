// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.
use criterion::{criterion_group, criterion_main, Criterion};

use argmin::core::{CostFunction, Error, Executor, Gradient, LineSearch};
use argmin::solver::linesearch::HagerZhangLineSearch;
use argmin_testfunctions::{sphere, sphere_derivative};

struct Sphere {}

impl CostFunction for Sphere {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        Ok(sphere(param))
    }
}

impl Gradient for Sphere {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, Error> {
        Ok(sphere_derivative(param))
    }
}

fn run() -> Result<(), Error> {
    // Define initial parameter vector
    let init_param: Vec<f64> = vec![1.0, 0.0];

    // Problem definition
    let operator = Sphere {};

    // Set up line search method
    let mut solver = HagerZhangLineSearch::new();

    // Set search direction
    solver.search_direction(vec![-1.5, -0.5]);
// Set initial step length
    solver.initial_step_length(10.0)?;

    let init_cost = operator.cost(&init_param)?;
    let init_grad = operator.gradient(&init_param)?;

    // Run solver
    let _res = Executor::new(operator, solver)
        // .add_observer(SlogLogger::term(), ObserverMode::Always)
        // Gradient and cost are optional. If they are not provided, they will be computed
        .configure(|state| {
            state
                .param(init_param)
                .gradient(init_grad)
                .cost(init_cost)
                .max_iters(100)
        })
        .run()?;
    Ok(())
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("HagerZhangLineSearch", |b| b.iter(run));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

