// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminConj;
use num_complex::Complex;

macro_rules! make_conj {
    ($t:ty) => {
        impl ArgminConj for Vec<$t> {
            #[inline]
            fn conj(&self) -> Vec<$t> {
                self.iter().map(|a| <$t as ArgminConj>::conj(a)).collect()
            }
        }

        impl ArgminConj for Vec<Vec<$t>> {
            #[inline]
            fn conj(&self) -> Vec<Vec<$t>> {
                self.iter()
                    .map(|a| a.iter().map(|b| <$t as ArgminConj>::conj(b)).collect())
                    .collect()
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
                fn [<test_conj_complex_vec_ $t>]() {
                    let a = vec![
                        Complex::new(1 as $t, 2 as $t),
                        Complex::new(4 as $t, -3 as $t),
                        Complex::new(8 as $t, 0 as $t)
                    ];
                    let b = vec![
                        Complex::new(1 as $t, -2 as $t),
                        Complex::new(4 as $t, 3 as $t),
                        Complex::new(8 as $t, 0 as $t)
                    ];
                    let res = <Vec<Complex<$t>> as ArgminConj>::conj(&a);
                    for i in 0..3 {
                        let tmp = b[i] - res[i];
                        let norm = ((tmp.re * tmp.re + tmp.im * tmp.im) as f64).sqrt();
                        assert!(norm  < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_conj_vec_ $t>]() {
                    let a = vec![1 as $t, 4 as $t, 8 as $t];
                    let b = vec![1 as $t, 4 as $t, 8 as $t];
                    let res = <Vec<$t> as ArgminConj>::conj(&a);
                    for i in 0..3 {
                        let diff = (b[i] as f64 - res[i] as f64).abs();
                        assert!(diff  < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_conj_complex_vec_vec_ $t>]() {
                    let a = vec![
                        vec![
                            Complex::new(1 as $t, 2 as $t),
                            Complex::new(4 as $t, -3 as $t),
                            Complex::new(8 as $t, 0 as $t)
                        ],
                        vec![
                            Complex::new(1 as $t, -5 as $t),
                            Complex::new(4 as $t, 6 as $t),
                            Complex::new(8 as $t, 0 as $t)
                        ],
                    ];
                    let b = vec![
                        vec![
                            Complex::new(1 as $t, -2 as $t),
                            Complex::new(4 as $t, 3 as $t),
                            Complex::new(8 as $t, 0 as $t)
                        ],
                        vec![
                            Complex::new(1 as $t, 5 as $t),
                            Complex::new(4 as $t, -6 as $t),
                            Complex::new(8 as $t, 0 as $t)
                        ],
                    ];
                    let res = <Vec<Vec<Complex<$t>>> as ArgminConj>::conj(&a);
                    for i in 0..2 {
                        for j in 0..3 {
                            let tmp = b[i][j] - res[i][j];
                            let norm = ((tmp.re * tmp.re + tmp.im * tmp.im) as f64).sqrt();
                            assert!(norm  < std::f64::EPSILON);
                        }
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
