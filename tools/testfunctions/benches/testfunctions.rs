// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

#![feature(test)]
#![feature(concat_idents)]

extern crate argmin_testfunctions;
extern crate test;
use argmin_testfunctions::*;

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

    make_bench!(ackley(&vec![-43.0, 53.0, 3.4]));
    make_bench!(ackley_param(
        &vec![-43.0, 53.0, 3.4],
        20.0,
        0.2,
        2.0 * ::std::f64::consts::PI
    ));

    make_bench!(beale(&vec![-4.0, 3.0]));

    make_bench!(booth(&vec![-4.0, 3.0]));

    make_bench!(bukin_n6(&vec![-4.0, 3.0]));

    make_bench!(cross_in_tray(&vec![-4.0, 3.0]));

    make_bench!(easom(&vec![-4.0, 3.0]));

    make_bench!(eggholder(&vec![-4.0, 3.0]));

    make_bench!(goldsteinprice(&vec![-4.0, 3.0]));

    make_bench!(himmelblau(&vec![-4.0, 3.0]));

    make_bench!(holder_table(&vec![-4.0, 3.0]));

    make_bench!(levy(&vec![-4.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0]));

    make_bench!(levy_n13(&vec![-4.0, 3.0]));

    make_bench!(matyas(&vec![-4.0, 3.0]));

    make_bench!(mccorminck(&vec![-4.0, 3.0]));

    make_bench!(picheny(&vec![0.5, 0.9]));

    make_bench!(rastrigin(&vec![-43.0, 53.0, 3.4]));
    make_bench!(rastrigin_a(&vec![-43.0, 53.0, 3.4], 10.0));

    make_bench!(rosenbrock(&vec![-43.0, 53.0, 3.4], 1_f64, 100_f64));
    make_bench!(rosenbrock_2d(&vec![-43.0, 53.0], 1_f64, 100_f64));
    make_bench!(rosenbrock_2d_derivative(&vec![-43.0, 53.0], 1_f64, 100_f64));
    make_bench!(rosenbrock_2d_hessian(&vec![-43.0, 53.0], 1_f64, 100_f64));

    make_bench!(sphere(&vec![-43.0, 53.0]));
    make_bench!(sphere_derivative(&vec![-43.0, 53.0]));

    make_bench!(styblinski_tang(&vec![-43.0, 53.0]));

    make_bench!(schaffer_n2(&vec![-43.0, 53.0]));
    make_bench!(schaffer_n4(&vec![-43.0, 53.0]));

    make_bench!(threehumpcamel(&vec![-43.0, 53.0]));

    make_bench!(zero(&vec![-43.0, 53.0]));
    make_bench!(zero_derivative(&vec![-43.0, 53.0]));
}
