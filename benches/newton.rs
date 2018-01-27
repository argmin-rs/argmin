// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

#![feature(test)]
#![allow(unused_imports)]

extern crate argmin;
extern crate ndarray;
extern crate test;
use ndarray::Array1;
use ndarray::prelude::*;
use argmin::ArgminSolver;
use argmin::problem::Problem;
use argmin::newton::Newton;
use argmin::testfunctions::{rosenbrock_derivative_nd, rosenbrock_hessian_nd, rosenbrock_nd};

#[cfg(test)]
mod tests {
    use ::*;
    use test::Bencher;

    #[bench]
    fn bench_newton_rosenbrock(b: &mut Bencher) {
        let cost = |x: &Array1<f64>| -> f64 { rosenbrock_nd(x, 1_f64, 100_f64) };
        let gradient =
            |x: &Array1<f64>| -> Array<f64, _> { rosenbrock_derivative_nd(x, 1_f64, 100_f64) };
        let hessian =
            |x: &Array1<f64>| -> Array<f64, _> { rosenbrock_hessian_nd(x, 1_f64, 100_f64) };

        let lower_bound: Array1<f64> = Array1::from_vec(vec![-1000.0, -1000.0]);
        let upper_bound: Array1<f64> = Array1::from_vec(vec![1000.0, 1000.0]);

        let mut prob = Problem::new(&cost, &lower_bound, &upper_bound);
        prob.gradient(&gradient);
        prob.hessian(&hessian);

        let mut solver = Newton::new();

        b.iter(|| {
            let init_param: Array1<f64> = prob.random_param().unwrap();
            solver.init(&prob, &init_param).unwrap();
            loop {
                if solver.next_iter().unwrap().iters >= 10 {
                    break;
                };
            }
        });
    }
}
