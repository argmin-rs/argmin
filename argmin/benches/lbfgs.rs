// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

use argmin::core::{CostFunction, Error, Executor, Gradient};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use argmin_testfunctions::{rosenbrock, rosenbrock_2d, rosenbrock_2d_derivative};
use finitediff::FiniteDiff;
use nalgebra::DVector;
use ndarray::{array, Array1};

struct RosenbrockVec {
    a: f64,
    b: f64,
}

struct RosenbrockNd {
    a: f64,
    b: f64,
}

impl CostFunction for RosenbrockVec {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        Ok(rosenbrock(&p.to_vec(), self.a, self.b))
    }
}

impl Gradient for RosenbrockVec {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
        Ok((*p).forward_diff(&|x| rosenbrock(&x, self.a, self.b)))
    }
}

impl CostFunction for RosenbrockNd {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        Ok(rosenbrock(&p.to_vec(), self.a, self.b))
    }
}

impl Gradient for RosenbrockNd {
    type Param = Array1<f64>;
    type Gradient = Array1<f64>;

    fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
        Ok((*p).forward_diff(&|x| rosenbrock(&x.to_vec(), self.a, self.b)))
    }
}

fn run_vec(
    a: f64,
    b: f64,
    init_param: &[f64],
    c1: f64,
    c2: f64,
    m: usize,
    iterations: u64,
) -> Result<(), Error> {
    // Define cost function
    let cost = RosenbrockVec { a, b };

    // Define initial parameter vector
    let init_param: Vec<f64> = Vec::from(init_param);
    // set up a line search
    let linesearch = MoreThuenteLineSearch::new().with_c(c1, c2)?;
    // Set up solver
    let solver = LBFGS::new(linesearch, m);

    // Run solver
    let res = Executor::new(cost, solver)
        .configure(|state| state.param(init_param).max_iters(iterations))
        .run()?;
    Ok(())
}


fn run_ndarray(
    a: f64,
    b: f64,
    init_param: &[f64],
    c1: f64,
    c2: f64,
    m: usize,
    iterations: u64,
) -> Result<(), Error> {
    // Define cost function
    let cost = RosenbrockNd { a, b };

    // Define initial parameter vector
    let init_param: Array1<f64> = Array1::from_vec(Vec::from(init_param));

    // set up a line search
    let linesearch = MoreThuenteLineSearch::new().with_c(c1, c2)?;
    // Set up solver
    let solver = LBFGS::new(linesearch, m);

    // Run solver
    let res = Executor::new(cost, solver)
        .configure(|state| state.param(init_param).max_iters(iterations))
        .run()?;
    Ok(())
}

fn criterion_benchmark(c: &mut Criterion) {
    let a = 1.0;
    let b = 100.0;
    let init_param = vec![-1.2, 1.0, -10.0, 2.0, 3.0, 2.0, 4.0, 10.0];
    let c1 = 1e-4;
    let c2 = 0.9;
    let m = 7;
    let iterations: u64 = 100;
    let mut group = c.benchmark_group("LBFGS");
    for i in 2..init_param.len() {
        group.bench_with_input(BenchmarkId::new("Vec", i), &i, |bencher, i| {
            bencher.iter(|| {
                run_vec(
                    black_box(a),
                    black_box(b),
                    black_box(&init_param[0..*i]),
                    black_box(c1),
                    black_box(c2),
                    black_box(m),
                    black_box(iterations),
                )
            })
        });
        group.bench_with_input(BenchmarkId::new("ndarray", i), &i, |bencher, i| {
            bencher.iter(|| {
                run_ndarray(
                    black_box(a),
                    black_box(b),
                    black_box(&init_param[0..*i]),
                    black_box(c1),
                    black_box(c2),
                    black_box(m),
                    black_box(iterations),
                )
            })
        });
    }
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
