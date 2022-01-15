// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminMul;
use ndarray::{Array1, Array2};
use num_complex::Complex;

macro_rules! make_mul {
    ($t:ty) => {
        impl ArgminMul<$t, Array1<$t>> for Array1<$t> {
            #[inline]
            fn mul(&self, other: &$t) -> Array1<$t> {
                self * *other
            }
        }

        impl ArgminMul<Array1<$t>, Array1<$t>> for $t {
            #[inline]
            fn mul(&self, other: &Array1<$t>) -> Array1<$t> {
                *self * other
            }
        }

        impl ArgminMul<Array1<$t>, Array1<$t>> for Array1<$t> {
            #[inline]
            fn mul(&self, other: &Array1<$t>) -> Array1<$t> {
                self * other
            }
        }

        impl ArgminMul<Array2<$t>, Array2<$t>> for Array2<$t> {
            #[inline]
            fn mul(&self, other: &Array2<$t>) -> Array2<$t> {
                self * other
            }
        }

        impl ArgminMul<$t, Array2<$t>> for Array2<$t> {
            #[inline]
            fn mul(&self, other: &$t) -> Array2<$t> {
                self * *other
            }
        }

        impl ArgminMul<Array2<$t>, Array2<$t>> for $t {
            #[inline]
            fn mul(&self, other: &Array2<$t>) -> Array2<$t> {
                *self * other
            }
        }
    };
}

macro_rules! make_complex_mul {
    ($t:ty) => {
        impl ArgminMul<Complex<$t>, Array1<Complex<$t>>> for Array1<Complex<$t>> {
            #[inline]
            fn mul(&self, other: &Complex<$t>) -> Array1<Complex<$t>> {
                self * *other
            }
        }

        impl ArgminMul<$t, Array1<Complex<$t>>> for Array1<Complex<$t>> {
            #[inline]
            fn mul(&self, other: &$t) -> Array1<Complex<$t>> {
                self * *other
            }
        }

        impl ArgminMul<Array1<Complex<$t>>, Array1<Complex<$t>>> for Complex<$t> {
            #[inline]
            fn mul(&self, other: &Array1<Complex<$t>>) -> Array1<Complex<$t>> {
                *self * other
            }
        }

        impl ArgminMul<Array1<Complex<$t>>, Array1<Complex<$t>>> for $t {
            #[inline]
            fn mul(&self, other: &Array1<Complex<$t>>) -> Array1<Complex<$t>> {
                Complex::new(*self, 0 as $t) * other
            }
        }

        impl ArgminMul<Array1<Complex<$t>>, Array1<Complex<$t>>> for Array1<Complex<$t>> {
            #[inline]
            fn mul(&self, other: &Array1<Complex<$t>>) -> Array1<Complex<$t>> {
                self * other
            }
        }

        impl ArgminMul<Array2<Complex<$t>>, Array2<Complex<$t>>> for Array2<Complex<$t>> {
            #[inline]
            fn mul(&self, other: &Array2<Complex<$t>>) -> Array2<Complex<$t>> {
                self * other
            }
        }

        impl ArgminMul<Complex<$t>, Array2<Complex<$t>>> for Array2<Complex<$t>> {
            #[inline]
            fn mul(&self, other: &Complex<$t>) -> Array2<Complex<$t>> {
                self * *other
            }
        }

        impl ArgminMul<$t, Array2<Complex<$t>>> for Array2<Complex<$t>> {
            #[inline]
            fn mul(&self, other: &$t) -> Array2<Complex<$t>> {
                self * *other
            }
        }

        impl ArgminMul<Array2<Complex<$t>>, Array2<Complex<$t>>> for Complex<$t> {
            #[inline]
            fn mul(&self, other: &Array2<Complex<$t>>) -> Array2<Complex<$t>> {
                *self * other
            }
        }
    };
}

make_mul!(i8);
make_mul!(u8);
make_mul!(i16);
make_mul!(u16);
make_mul!(i32);
make_mul!(u32);
make_mul!(i64);
make_mul!(u64);
make_mul!(f32);
make_mul!(f64);
make_complex_mul!(f32);
make_complex_mul!(f64);

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_mul_vec_scalar_ $t>]() {
                    let a = array![1 as $t, 4 as $t, 8 as $t];
                    let b = 2 as $t;
                    let target = array![2 as $t, 8 as $t, 16 as $t];
                    let res = <Array1<$t> as ArgminMul<$t, Array1<$t>>>::mul(&a, &b);
                    for i in 0..3 {
                        assert!(((target[i] - res[i]) as f64).abs() < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mul_scalar_vec_ $t>]() {
                    let a = array![1 as $t, 4 as $t, 8 as $t];
                    let b = 2 as $t;
                    let target = array![2 as $t, 8 as $t, 16 as $t];
                    let res = <$t as ArgminMul<Array1<$t>, Array1<$t>>>::mul(&b, &a);
                    for i in 0..3 {
                        assert!(((target[i] - res[i]) as f64).abs() < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mul_vec_vec_ $t>]() {
                    let a = array![1 as $t, 4 as $t, 8 as $t];
                    let b = array![2 as $t, 3 as $t, 4 as $t];
                    let target = array![2 as $t, 12 as $t, 32 as $t];
                    let res = <Array1<$t> as ArgminMul<Array1<$t>, Array1<$t>>>::mul(&a, &b);
                    for i in 0..3 {
                        assert!(((target[i] - res[i]) as f64).abs() < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_mul_vec_vec_panic_ $t>]() {
                    let a = array![1 as $t, 4 as $t];
                    let b = array![41 as $t, 38 as $t, 34 as $t];
                    <Array1<$t> as ArgminMul<Array1<$t>, Array1<$t>>>::mul(&a, &b);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_mul_vec_vec_panic_2_ $t>]() {
                    let a = array![];
                    let b = array![41 as $t, 38 as $t, 34 as $t];
                    <Array1<$t> as ArgminMul<Array1<$t>, Array1<$t>>>::mul(&a, &b);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_mul_vec_vec_panic_3_ $t>]() {
                    let a = array![41 as $t, 38 as $t, 34 as $t];
                    let b = array![];
                    <Array1<$t> as ArgminMul<Array1<$t>, Array1<$t>>>::mul(&a, &b);
                }
            }

            item! {
                #[test]
                fn [<test_mul_mat_mat_ $t>]() {
                    let a = array![
                        [1 as $t, 4 as $t, 8 as $t],
                        [2 as $t, 5 as $t, 9 as $t]
                    ];
                    let b = array![
                        [2 as $t, 3 as $t, 4 as $t],
                        [3 as $t, 4 as $t, 5 as $t]
                    ];
                    let target = array![
                        [2 as $t, 12 as $t, 32 as $t],
                        [6 as $t, 20 as $t, 45 as $t]
                    ];
                    let res = <Array2<$t> as ArgminMul<Array2<$t>, Array2<$t>>>::mul(&a, &b);
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
                fn [<test_mul_mat_mat_panic_2_ $t>]() {
                    let a = array![
                        [1 as $t, 4 as $t, 8 as $t],
                        [2 as $t, 5 as $t, 9 as $t]
                    ];
                    let b = array![
                        [41 as $t, 38 as $t],
                    ];
                    <Array2<$t> as ArgminMul<Array2<$t>, Array2<$t>>>::mul(&a, &b);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_mul_mat_mat_panic_3_ $t>]() {
                    let a = array![
                        [1 as $t, 4 as $t, 8 as $t],
                        [2 as $t, 5 as $t, 9 as $t]
                    ];
                    let b = array![[]];
                    <Array2<$t> as ArgminMul<Array2<$t>, Array2<$t>>>::mul(&a, &b);
                }
            }

            item! {
                #[test]
                fn [<test_mul_scalar_mat_1_ $t>]() {
                    let a = array![
                        [1 as $t, 4 as $t, 8 as $t],
                        [2 as $t, 5 as $t, 9 as $t]
                    ];
                    let b = 2 as $t;
                    let target = array![
                        [2 as $t, 8 as $t, 16 as $t],
                        [4 as $t, 10 as $t, 18 as $t]
                    ];
                    let res = <Array2<$t> as ArgminMul<$t, Array2<$t>>>::mul(&a, &b);
                    for i in 0..3 {
                        for j in 0..2 {
                        assert!(((target[(j, i)] - res[(j, i)]) as f64).abs() < std::f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mul_scalar_mat_2_ $t>]() {
                    let b = array![
                        [1 as $t, 4 as $t, 8 as $t],
                        [2 as $t, 5 as $t, 9 as $t]
                    ];
                    let a = 2 as $t;
                    let target = array![
                        [2 as $t, 8 as $t, 16 as $t],
                        [4 as $t, 10 as $t, 18 as $t]
                    ];
                    let res = <$t as ArgminMul<Array2<$t>, Array2<$t>>>::mul(&a, &b);
                    for i in 0..3 {
                        for j in 0..2 {
                        assert!(((target[(j, i)] - res[(j, i)]) as f64).abs() < std::f64::EPSILON);
                        }
                    }
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
