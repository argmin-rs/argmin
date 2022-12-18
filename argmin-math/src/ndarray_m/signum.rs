// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminSignum;
use ndarray::{Array1, Array2};
use num_complex::Complex;

macro_rules! make_signum {
    ($t:ty) => {
        impl ArgminSignum for Array1<$t> {
            #[inline]
            fn signum(mut self) -> Array1<$t> {
                for a in &mut self {
                    *a = a.signum();
                }
                self
            }
        }

        impl ArgminSignum for Array2<$t> {
            #[inline]
            fn signum(mut self) -> Array2<$t> {
                let m = self.shape()[0];
                let n = self.shape()[1];
                for i in 0..m {
                    for j in 0..n {
                        self[(i, j)] = self[(i, j)].signum();
                    }
                }
                self
            }
        }
    };
}

macro_rules! make_signum_complex {
    ($t:ty) => {
        impl ArgminSignum for Array1<$t> {
            #[inline]
            fn signum(mut self) -> Array1<$t> {
                for a in &mut self {
                    a.re = a.re.signum();
                    a.im = a.im.signum();
                }
                self
            }
        }

        impl ArgminSignum for Array2<$t> {
            #[inline]
            fn signum(mut self) -> Array2<$t> {
                let m = self.shape()[0];
                let n = self.shape()[1];
                for i in 0..m {
                    for j in 0..n {
                        self[(i, j)].re = self[(i, j)].re.signum();
                        self[(i, j)].im = self[(i, j)].im.signum();
                    }
                }
                self
            }
        }
    };
}

make_signum!(isize);
make_signum!(i8);
make_signum!(i16);
make_signum!(i32);
make_signum!(i64);
make_signum!(f32);
make_signum!(f64);
make_signum_complex!(Complex<isize>);
make_signum_complex!(Complex<i8>);
make_signum_complex!(Complex<i16>);
make_signum_complex!(Complex<i32>);
make_signum_complex!(Complex<i64>);
make_signum_complex!(Complex<f32>);
make_signum_complex!(Complex<f64>);

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_signum_complex_array1_ $t>]() {
                    let x = Array1::from(vec![
                        Complex::new(1 as $t, 2 as $t),
                        Complex::new(4 as $t, -3 as $t),
                        Complex::new(-8 as $t, 4 as $t),
                        Complex::new(-8 as $t, -1 as $t),
                    ]);
                    let y = Array1::from(vec![
                        Complex::new(1 as $t, 1 as $t),
                        Complex::new(1 as $t, -1 as $t),
                        Complex::new(-1 as $t, 1 as $t),
                        Complex::new(-1 as $t, -1 as $t),
                    ]);
                    let res = <Array1<Complex<$t>> as ArgminSignum>::signum(x);
                    for i in 0..4 {
                        let tmp = y[i] - res[i];
                        let norm = ((tmp.re * tmp.re + tmp.im * tmp.im) as f64).sqrt();
                        assert!(norm < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_signum_array1_ $t>]() {
                    let x = Array1::from(vec![1 as $t, -4 as $t, 8 as $t]);
                    let y = Array1::from(vec![1 as $t, -1 as $t, 1 as $t]);
                    let res = <Array1<$t> as ArgminSignum>::signum(x);
                    for i in 0..3 {
                        let diff = (y[i] - res[i]).abs() as f64;
                        assert!(diff < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_signum_complex_array2_ $t>]() {
                    let x = array![
                        [
                            Complex::new(1 as $t, 2 as $t),
                            Complex::new(4 as $t, -3 as $t),
                            Complex::new(-8 as $t, 4 as $t),
                            Complex::new(-8 as $t, -1 as $t),
                        ],
                        [
                            Complex::new(-1 as $t, -2 as $t),
                            Complex::new(4 as $t, 3 as $t),
                            Complex::new(-8 as $t, -4 as $t),
                            Complex::new(8 as $t, -1 as $t),
                        ]
                    ];
                    let y = array![
                        [
                            Complex::new(1 as $t, 1 as $t),
                            Complex::new(1 as $t, -1 as $t),
                            Complex::new(-1 as $t, 1 as $t),
                            Complex::new(-1 as $t, -1 as $t),
                        ],
                        [
                            Complex::new(-1 as $t, -1 as $t),
                            Complex::new(1 as $t, 1 as $t),
                            Complex::new(-1 as $t, -1 as $t),
                            Complex::new(1 as $t, -1 as $t),
                        ]
                    ];
                    let res = <Array2<Complex<$t>> as ArgminSignum>::signum(x);
                    for j in 0..2 {
                        for i in 0..4 {
                            let tmp = y[(j, i)] - res[(j, i)];
                            let norm = ((tmp.re * tmp.re + tmp.im * tmp.im) as f64).sqrt();
                            assert!(norm < std::f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_signum_array2_ $t>]() {
                    let x = array![
                        [1 as $t, -4 as $t, 8 as $t],
                        [-2 as $t, 2 as $t, -8 as $t]
                    ];
                    let y = array![
                        [1 as $t, -1 as $t, 1 as $t],
                        [-1 as $t, 1 as $t, -1 as $t]
                    ];
                    let res = <Array2<$t> as ArgminSignum>::signum(x);
                    for j in 0..2 {
                        for i in 0..3 {
                            let diff = (y[(j, i)] - res[(j, i)]).abs() as f64;
                            assert!(diff < std::f64::EPSILON);
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
