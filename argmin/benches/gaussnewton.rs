// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.
use criterion::{black_box, criterion_group, criterion_main, Criterion};

use argmin::core::{Error, Executor, Jacobian, Operator};
use argmin::solver::gaussnewton::GaussNewton;
use ndarray::{Array1, Array2};
use nalgebra::{DVector, DMatrix};

type Rate = f64;
type S = f64;
type Measurement = (S, Rate);

// Example taken from Wikipedia: https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm
// Model used in this example:
// `rate = (V_{max} * [S]) / (K_M + [S]) `
// where `V_{max}` and `K_M` are the sought parameters and `[S]` and `rate` is the measured data.

struct ProblemNG {
    data: Vec<Measurement>,
}

struct ProblemNd {
    data: Vec<Measurement>,
}



impl Operator for ProblemNd {
    type Param = Array1<f64>;
    type Output = Array1<f64>;

    fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        Ok(self
            .data
            .iter()
            .map(|(s, rate)| rate - (p[0] * s) / (p[1] + s))
            .collect::<Array1<f64>>())
    }
}

impl Operator for ProblemNG {
    type Param = DVector<f64>;
    type Output = DVector<f64>;

    fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        Ok(DVector::from_vec(
            self.data
                .iter()
                .map(|(s, rate)| rate - (p[0] * s) / (p[1] + s))
                .collect(),
        ))
    }
}


impl Jacobian for ProblemNd {
    type Param = Array1<f64>;
    type Jacobian = Array2<f64>;

    fn jacobian(&self, p: &Self::Param) -> Result<Self::Jacobian, Error> {
        Ok(Array2::from_shape_fn((self.data.len(), 2), |(si, i)| {
            if i == 0 {
                -self.data[si].0 / (p[1] + self.data[si].0)
            } else {
                p[0] * self.data[si].0 / (p[1] + self.data[si].0).powi(2)
            }
        }))
    }
}


impl Jacobian for ProblemNG {
    type Param = DVector<f64>;
    type Jacobian = DMatrix<f64>;

    fn jacobian(&self, p: &Self::Param) -> Result<Self::Jacobian, Error> {
        Ok(DMatrix::from_fn(7, 2, |si, i| {
            if i == 0 {
                -self.data[si].0 / (p[1] + self.data[si].0)
            } else {
                p[0] * self.data[si].0 / (p[1] + self.data[si].0).powi(2)
            }
        }))
    }
}


fn run_ngalgebra(data: &Vec<(f64, f64)>, init_param: (f64,f64), iterations: u64) -> Result<(), Error> {
    // Define cost function
    // Example taken from Wikipedia: https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm
    let cost = ProblemNG {
        data: data.clone()
    };

    // Define initial parameter vector
    let init_param: DVector<f64> = DVector::from_vec(vec![init_param.0, init_param.1]);

    // Set up solver
    let solver: GaussNewton<f64> = GaussNewton::new();

    // Run solver
    let res = Executor::new(cost, solver)
        .configure(|state| state.param(init_param).max_iters(iterations))
        .run()?;
    Ok(())
}


fn run_ndarray(data: &Vec<(f64, f64)>, init_param: (f64,f64), iterations: u64) -> Result<(), Error> {
    // Define cost function
    // Example taken from Wikipedia: https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm
    let cost = ProblemNd {
        data: data.clone(),
    };
    // Define initial parameter vector
    let init_param: Array1<f64> = Array1::from(vec![init_param.0, init_param.1]);
    // Set up solver
    let solver: GaussNewton<f64> = GaussNewton::new();

    // Run solver
    let res = Executor::new(cost, solver)
        .configure(|state| state.param(init_param).max_iters(iterations))
        .run()?;
    Ok(())
}

fn criterion_benchmark(c: &mut Criterion) {
    let data = vec![
        (0.038, 0.050),
        (0.194, 0.127),
        (0.425, 0.094),
        (0.626, 0.2122),
        (1.253, 0.2729),
        (2.5, 0.2665),
        (3.74, 0.3317),
    ];
    let init_param = (0.9, 0.2);
    let iterations = 10;
    let mut group = c.benchmark_group("GaussNewton");
    group.bench_function("GaussNewton_ngalgebra", |b| {
        b.iter(|| {
            run_ngalgebra(
                black_box(&data),
                black_box(init_param),
                black_box(iterations),
            )
            .expect("Benchmark should run without errors")
        })
    });
    group.bench_function("GaussNewton_ndarry", |b| {
        b.iter(|| {
            run_ndarray(
                black_box(&data),
                black_box(init_param),
                black_box(iterations),
            )
            .expect("Benchmark should run without errors")
        })
    });
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
