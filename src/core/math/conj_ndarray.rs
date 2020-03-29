// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

// TODO: Tests for Array2 impl

use crate::core::math::ArgminConj;
use ndarray::{Array1, Array2};
use num_complex::Complex;

macro_rules! make_conj {
    ($t:ty) => {
        impl ArgminConj for Array1<$t> {
            #[inline]
            fn conj(&self) -> Array1<$t> {
                self.iter().map(|a| <$t as ArgminConj>::conj(a)).collect()
            }
        }

        impl ArgminConj for Array2<$t> {
            #[inline]
            fn conj(&self) -> Array2<$t> {
                let n = self.shape();
                let mut out = self.clone();
                for i in 0..n[0] {
                    for j in 0..n[1] {
                        out[(i, j)] = out[(i, j)].conj();
                    }
                }
                out
            }
        }
    };
}

make_conj!(isize);
make_conj!(i8);
make_conj!(i16);
make_conj!(i32);
make_conj!(i64);
make_conj!(f32);
make_conj!(f64);
make_conj!(Complex<isize>);
make_conj!(Complex<i8>);
make_conj!(Complex<i16>);
make_conj!(Complex<i32>);
make_conj!(Complex<i64>);
make_conj!(Complex<f32>);
make_conj!(Complex<f64>);

#[cfg(test)]
mod tests {
    use super::*;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_conj_complex_ndarray_ $t>]() {
                    let a = Array1::from(vec![
                        Complex::new(1 as $t, 2 as $t),
                        Complex::new(4 as $t, -3 as $t),
                        Complex::new(8 as $t, 0 as $t)
                    ]);
                    let b = vec![
                        Complex::new(1 as $t, -2 as $t),
                        Complex::new(4 as $t, 3 as $t),
                        Complex::new(8 as $t, 0 as $t)
                    ];
                    let res = <Array1<Complex<$t>> as ArgminConj>::conj(&a);
                    for i in 0..3 {
                        let tmp = b[i] - res[i];
                        let norm = ((tmp.re * tmp.re + tmp.im * tmp.im) as f64).sqrt();
                        assert!(norm  < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_conj_ndarray_ $t>]() {
                    let a = vec![1 as $t, 4 as $t, 8 as $t];
                    let b = vec![1 as $t, 4 as $t, 8 as $t];
                    let res = <Vec<$t> as ArgminConj>::conj(&a);
                    for i in 0..3 {
                        let diff = (b[i] as f64 - res[i] as f64).abs();
                        assert!(diff  < std::f64::EPSILON);
                    }
                }
            }
        };
    }

    make_test!(isize);
    make_test!(i8);
    make_test!(i16);
    make_test!(i32);
    make_test!(i64);
    make_test!(f32);
    make_test!(f64);
}
