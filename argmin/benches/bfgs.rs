// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use argmin::core::{CostFunction, Error, Executor, Gradient};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::BFGS;
use argmin_testfunctions::rosenbrock;
use finitediff::FiniteDiff;
use nalgebra::uninit::InitStatus;
use ndarray::{array, Array1, Array2, FixedInitializer};

struct RosenbrockVec {
    a: f64,
    b: f64,
}

struct RosenbrockNdarray {
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

impl CostFunction for RosenbrockNdarray {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        Ok(rosenbrock(&p.to_vec(), self.a, self.b))
    }
}

impl Gradient for RosenbrockNdarray {
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
    iterations: u64,
) -> Result<(), Error> {
    // Define cost function
    let cost = RosenbrockVec { a, b };
    // Define initial parameter vector
    let init_param: Vec<f64> = Vec::from(init_param);
    let mut init_hessian = Vec::<Vec<f64>>::new();
    for i in 0..init_hessian.len() {
        let mut row = Vec::new();
        for j in 0..init_hessian.len() {
            if i == j {
                row.push(1.0);
            } else {
                row.push(0.0);
            }
        }
        init_hessian.push(row);
    }
    // set up a line search
    let linesearch = MoreThuenteLineSearch::new().with_c(c1, c2)?;
    // Set up solver
    let solver = BFGS::new(linesearch);

    // Run solver
    let res = Executor::new(cost, solver)
        .configure(|state| {
            state
                .param(init_param)
                .inv_hessian(init_hessian)
                .max_iters(iterations)
        })
        .run()?;
    Ok(())
}
fn run_ndarray(
    a: f64,
    b: f64,
    init_param: &[f64],
    c1: f64,
    c2: f64,
    iterations: u64,
) -> Result<(), Error> {
    // Define cost function
    let cost = RosenbrockNdarray { a, b };
    // Define initial parameter vector
    let init_param: Array1<f64> = Array1::from_vec(Vec::from(init_param));
    let init_hessian: Array2<f64> = Array2::eye(init_param.len());
    // set up a line search
    let linesearch = MoreThuenteLineSearch::new().with_c(c1, c2)?;
    // Set up solver
    let solver = BFGS::new(linesearch);

    // Run solver
    let res = Executor::new(cost, solver)
        .configure(|state| {
            state
                .param(init_param)
                .inv_hessian(init_hessian)
                .max_iters(iterations)
        })
        .run()?;
    Ok(())
}

fn criterion_benchmark(c: &mut Criterion) {
    let a = 1.0;
    let b = 100.0;
    let init_param = vec![-1.2, 1.0, -10.0, 2.0, 3.0, 2.0];
    let c1 = 1e-4;
    let c2 = 0.9;
    let iterations: u64 = 60;
    let mut group = c.benchmark_group("BFGS");
    for i in 2..init_param.len() {
        // WARN: Vec version immediately fails with
        // Condition violated: `MoreThuenteLineSearch`: Search direction must be a descent direction.
        //
        // group.bench_with_input(BenchmarkId::new("Vec", i), &i, |bencher, i| {
        //     bencher.iter(|| {
        //         run_vec(
        //             black_box(a),
        //             black_box(b),
        //             black_box(&init_param[0..*i]),
        //             black_box(c1),
        //             black_box(c2),
        //             black_box(iterations),
        //         ).expect("Benchmark should run without errors")
        //     })
        // });
        group.bench_with_input(BenchmarkId::new("ndarray", i), &i, |bencher, i| {
            bencher.iter(|| {
                run_ndarray(
                    black_box(a),
                    black_box(b),
                    black_box(&init_param[0..*i]),
                    black_box(c1),
                    black_box(c2),
                    black_box(iterations),
                )
                .expect("Benchmark should run without errors")
            })
        });
    }
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
