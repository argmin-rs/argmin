// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

#![feature(test)]

extern crate argmin_testfunctions;
extern crate test;
use paste::item;

macro_rules! make_bench {
    ($f:ident($p:expr)) => {
        item! {
            #[bench]
            fn [<bench_ $f>](b: &mut Bencher) {
                b.iter(|| {
                    let params = $p;
                    // black_box(::$f($($x),*));
                    black_box($f(params));
                });
            }
        }
    };

    ($f:ident($p:expr, $a:expr)) => {
        item! {
            #[bench]
            fn [<bench_ $f>](b: &mut Bencher) {
                b.iter(|| {
                    let params = $p;
                    let a = $a;
                    black_box($f(params, a));
                });
            }
        }
    };

    ($f:ident($p:expr, $a:expr, $b:expr)) => {
        item! {
            #[bench]
            fn [<bench_ $f>](b: &mut Bencher) {
                b.iter(|| {
                    let params = $p;
                    let a = $a;
                    let b = $b;
                    black_box($f(params, a, b));
                });
            }
        }
    };

    ($f:ident($p:expr, $a:expr, $b:expr, $c:expr)) => {
        item! {
            #[bench]
            fn [<bench_ $f>](b: &mut Bencher) {
                b.iter(|| {
                    let params = $p;
                    let a = $a;
                    let b = $b;
                    let c = $b;
                    black_box($f(params, a, b, c));
                });
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use argmin_testfunctions::*;
    use test::{black_box, Bencher};

    make_bench!(ackley(&[-43.0, 53.0, 3.4]));
    make_bench!(ackley_abc(
        &[-43.0, 53.0, 3.4],
        20.0,
        0.2,
        2.0 * ::std::f64::consts::PI
    ));

    make_bench!(beale(&[-4.0, 3.0]));

    make_bench!(booth(&[-4.0, 3.0]));

    make_bench!(bukin_n6(&[-4.0, 3.0]));

    make_bench!(cross_in_tray(&[-4.0, 3.0]));

    make_bench!(easom(&[-4.0, 3.0]));

    make_bench!(eggholder(&[-4.0, 3.0]));

    make_bench!(goldsteinprice(&[-4.0, 3.0]));

    make_bench!(himmelblau(&[-4.0, 3.0]));

    make_bench!(holder_table(&[-4.0, 3.0]));

    make_bench!(levy(&[-4.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0]));

    make_bench!(levy_n13(&[-4.0, 3.0]));

    make_bench!(matyas(&[-4.0, 3.0]));

    make_bench!(mccorminck(&[-4.0, 3.0]));

    make_bench!(picheny(&[0.5, 0.9]));

    make_bench!(rastrigin(&[-43.0, 53.0, 3.4]));
    make_bench!(rastrigin_a(&[-43.0, 53.0, 3.4], 10.0));

    make_bench!(rosenbrock(&[-43.0, 53.0, 3.4], 1_f64, 100_f64));
    make_bench!(rosenbrock_derivative(&[-43.0, 53.0], 1_f64, 100_f64));
    make_bench!(rosenbrock_derivative_const(&[-43.0, 53.0], 1_f64, 100_f64));
    make_bench!(rosenbrock_hessian(&[-43.0, 53.0], 1_f64, 100_f64));
    make_bench!(rosenbrock_hessian_const(&[-43.0, 53.0], 1_f64, 100_f64));

    make_bench!(sphere(&vec![-43.0, 53.0]));
    make_bench!(sphere_derivative(&[-43.0, 53.0]));

    make_bench!(styblinski_tang(&[-43.0, 53.0]));

    make_bench!(schaffer_n2(&[-43.0, 53.0]));
    make_bench!(schaffer_n4(&[-43.0, 53.0]));

    make_bench!(threehumpcamel(&[-43.0, 53.0]));

    make_bench!(zero(&[-43.0, 53.0]));
    make_bench!(zero_derivative(&[-43.0, 53.0]));
}
