// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.
use criterion::{black_box, criterion_group, criterion_main, Criterion};

use argmin::core::{CostFunction, Error, Executor};
use argmin::solver::particleswarm::ParticleSwarm;
use argmin_testfunctions::himmelblau;
use nalgebra::{dvector, DVector};
use ndarray::{array, Array1};

struct HimmelblauVec {}

impl CostFunction for HimmelblauVec {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        Ok(himmelblau(param))
    }
}

struct HimmelblauNG {}

impl CostFunction for HimmelblauNG {
    type Param = DVector<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        Ok(himmelblau(param.into()))
    }
}

struct HimmelblauNdarray {}

impl CostFunction for HimmelblauNdarray {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        Ok(himmelblau(param.as_slice().unwrap()))
    }
}

fn run_vec(bound: f64, num_particles: usize, iterations: u64) -> Result<(), Error> {
    let cost_function = HimmelblauVec {};

    let solver = ParticleSwarm::new((vec![-bound, -bound], vec![bound, bound]), num_particles);

    let res = Executor::new(cost_function, solver)
        .configure(|state| state.max_iters(iterations))
        .run()?;
    Ok(())
}

fn run_ngalgebra(bound: f64, num_particles: usize, iterations: u64) -> Result<(), Error> {
    let cost_function = HimmelblauNG {};

    let solver = ParticleSwarm::new(
        (dvector![-bound, -bound], dvector![bound, bound]),
        num_particles,
    );

    let res = Executor::new(cost_function, solver)
        .configure(|state| state.max_iters(iterations))
        .run()?;
    Ok(())
}

fn run_ndarray(bound: f64, num_particles: usize, iterations: u64) -> Result<(), Error> {
    let cost_function = HimmelblauNdarray {};

    let solver = ParticleSwarm::new(
        (array![-bound, -bound], array![bound, bound]),
        num_particles,
    );

    let res = Executor::new(cost_function, solver)
        .configure(|state| state.max_iters(iterations))
        .run()?;
    Ok(())
}

fn criterion_benchmark(c: &mut Criterion) {
    let bound = 4.0;
    let num_particles = 40;
    let iterations = 100;
    let mut group = c.benchmark_group("ParticleSwarm");
    group.bench_function("ParticleSwarm_Vec", |b| {
        b.iter(|| run_vec(black_box(bound), black_box(num_particles), black_box(iterations)))
    });
    group.bench_function("ParticleSwarm_ngalgebra", |b| {
        b.iter(|| run_ngalgebra(black_box(bound), black_box(num_particles), black_box(iterations)))
    });
    group.bench_function("ParticleSwarm_ndarry", |b| {
        b.iter(|| run_ndarray(black_box(bound), black_box(num_particles), black_box(iterations)))
    });
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
