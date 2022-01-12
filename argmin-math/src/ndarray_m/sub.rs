// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminSub;
use ndarray::{Array1, Array2};
use num_complex::Complex;

macro_rules! make_sub {
    ($t:ty) => {
        impl ArgminSub<$t, Array1<$t>> for Array1<$t> {
            #[inline]
            fn sub(&self, other: &$t) -> Array1<$t> {
                self - *other
            }
        }

        impl ArgminSub<Array1<$t>, Array1<$t>> for $t {
            #[inline]
            fn sub(&self, other: &Array1<$t>) -> Array1<$t> {
                *self - other
            }
        }

        impl ArgminSub<Array1<$t>, Array1<$t>> for Array1<$t> {
            #[inline]
            fn sub(&self, other: &Array1<$t>) -> Array1<$t> {
                self - other
            }
        }

        impl ArgminSub<Array2<$t>, Array2<$t>> for Array2<$t> {
            #[inline]
            fn sub(&self, other: &Array2<$t>) -> Array2<$t> {
                self - other
            }
        }

        impl ArgminSub<$t, Array2<$t>> for Array2<$t> {
            #[inline]
            fn sub(&self, other: &$t) -> Array2<$t> {
                self - *other
            }
        }
    };
}

make_sub!(i8);
make_sub!(i16);
make_sub!(i32);
make_sub!(i64);
make_sub!(u8);
make_sub!(u16);
make_sub!(u32);
make_sub!(u64);
make_sub!(f32);
make_sub!(f64);
make_sub!(Complex<f32>);
make_sub!(Complex<f64>);

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_sub_vec_scalar_ $t>]() {
                    let a = array![36 as $t, 39 as $t, 43 as $t];
                    let b = 1 as $t;
                    let target = array![35 as $t, 38 as $t, 42 as $t];
                    let res = <Array1<$t> as ArgminSub<$t, Array1<$t>>>::sub(&a, &b);
                    for i in 0..3 {
                        assert!(((target[i] - res[i]) as f64).abs() < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_sub_scalar_vec_ $t>]() {
                    let a = array![1 as $t, 4 as $t, 8 as $t];
                    let b = 34 as $t;
                    let target = array![33 as $t, 30 as $t, 26 as $t];
                    let res = <$t as ArgminSub<Array1<$t>, Array1<$t>>>::sub(&b, &a);
                    for i in 0..3 {
                        assert!(((target[i] - res[i]) as f64).abs() < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_sub_vec_vec_ $t>]() {
                    let a = array![41 as $t, 38 as $t, 34 as $t];
                    let b = array![1 as $t, 4 as $t, 8 as $t];
                    let target = array![40 as $t, 34 as $t, 26 as $t];
                    let res = <Array1<$t> as ArgminSub<Array1<$t>, Array1<$t>>>::sub(&a, &b);
                    for i in 0..3 {
                        assert!(((target[i] - res[i]) as f64).abs() < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_sub_vec_vec_panic_ $t>]() {
                    let a = array![41 as $t, 38 as $t, 34 as $t];
                    let b = array![1 as $t, 4 as $t];
                    <Array1<$t> as ArgminSub<Array1<$t>, Array1<$t>>>::sub(&a, &b);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_sub_vec_vec_panic_2_ $t>]() {
                    let a = array![];
                    let b = array![41 as $t, 38 as $t, 34 as $t];
                    <Array1<$t> as ArgminSub<Array1<$t>, Array1<$t>>>::sub(&a, &b);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_sub_vec_vec_panic_3_ $t>]() {
                    let a = array![41 as $t, 38 as $t, 34 as $t];
                    let b = array![];
                    <Array1<$t> as ArgminSub<Array1<$t>, Array1<$t>>>::sub(&a, &b);
                }
            }

            item! {
                #[test]
                fn [<test_sub_mat_mat_ $t>]() {
                    let a = array![
                        [43 as $t, 46 as $t, 50 as $t],
                        [44 as $t, 47 as $t, 51 as $t]
                    ];
                    let b = array![
                        [1 as $t, 4 as $t, 8 as $t],
                        [2 as $t, 5 as $t, 9 as $t]
                    ];
                    let target = array![
                        [42 as $t, 42 as $t, 42 as $t],
                        [42 as $t, 42 as $t, 42 as $t]
                    ];
                    let res = <Array2<$t> as ArgminSub<Array2<$t>, Array2<$t>>>::sub(&a, &b);
                    for i in 0..3 {
                        for j in 0..2 {
                        assert!(((target[(j, i)] - res[(j, i)]) as f64).abs() < std::f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_sub_mat_scalar_ $t>]() {
                    let a = array![
                        [43 as $t, 46 as $t, 50 as $t],
                        [44 as $t, 47 as $t, 51 as $t]
                    ];
                    let b = 2 as $t;
                    let target = array![
                        [41 as $t, 44 as $t, 48 as $t],
                        [42 as $t, 45 as $t, 49 as $t]
                    ];
                    let res = <Array2<$t> as ArgminSub<$t, Array2<$t>>>::sub(&a, &b);
                    for i in 0..3 {
                        for j in 0..2 {
                        assert!(((target[(j, i)] - res[(j, i)]) as f64).abs() < std::f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_sub_mat_mat_panic_2_ $t>]() {
                    let a = array![
                        [41 as $t, 38 as $t],
                    ];
                    let b = array![
                        [1 as $t, 4 as $t, 8 as $t],
                        [2 as $t, 5 as $t, 9 as $t]
                    ];
                    <Array2<$t> as ArgminSub<Array2<$t>, Array2<$t>>>::sub(&a, &b);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_sub_mat_mat_panic_3_ $t>]() {
                    let a = array![
                        [1 as $t, 4 as $t, 8 as $t],
                        [2 as $t, 5 as $t, 9 as $t]
                    ];
                    let b = array![[]];
                    <Array2<$t> as ArgminSub<Array2<$t>, Array2<$t>>>::sub(&a, &b);
                }
            }
        };
    }

    make_test!(i8);
    make_test!(u8);
    make_test!(i16);
    make_test!(u16);
    make_test!(i32);
    make_test!(u32);
    make_test!(i64);
    make_test!(u64);
    make_test!(f32);
    make_test!(f64);
}
