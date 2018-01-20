#![feature(test)]
#![feature(macro_rules)]
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
