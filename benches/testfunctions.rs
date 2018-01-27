// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

#![feature(test)]
#![feature(concat_idents)]

extern crate argmin;
extern crate ndarray;
extern crate test;
use ndarray::Array1;
use argmin::testfunctions::*;

macro_rules! make_bench {
    ($f:ident($($x:expr),*)) => {
        #[bench]
        fn $f(b: &mut Bencher) {
            b.iter(|| {
                black_box(::$f($($x),*));
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use test::{black_box, Bencher};

    make_bench!(rosenbrock_nd(
        &::Array1::from_vec(vec![-43.0, 53.0]),
        1_f64,
        100_f64
    ));

    make_bench!(rosenbrock_derivative_nd(
        &::Array1::from_vec(vec![-43.0, 53.0]),
        1_f64,
        100_f64
    ));

    make_bench!(rosenbrock_hessian_nd(
        &::Array1::from_vec(vec![-43.0, 53.0]),
        1_f64,
        100_f64
    ));

    make_bench!(rosenbrock(&vec![-43.0, 53.0], 1_f64, 100_f64));
    make_bench!(rosenbrock_derivative(&vec![-43.0, 53.0], 1_f64, 100_f64));
    make_bench!(rosenbrock_hessian(&vec![-43.0, 53.0], 1_f64, 100_f64));

    make_bench!(sphere(&vec![-43.0, 53.0]));
    make_bench!(sphere_derivative(&vec![-43.0, 53.0]));
}
