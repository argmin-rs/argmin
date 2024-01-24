// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminSub;
use num_complex::Complex;

macro_rules! make_sub {
    ($t:ty) => {
        impl ArgminSub<$t, Vec<$t>> for Vec<$t> {
            #[inline]
            fn sub(&self, other: &$t) -> Vec<$t> {
                self.iter().map(|a| a - other).collect()
            }
        }

        impl ArgminSub<Vec<$t>, Vec<$t>> for $t {
            #[inline]
            fn sub(&self, other: &Vec<$t>) -> Vec<$t> {
                other.iter().map(|a| self - a).collect()
            }
        }

        impl ArgminSub<Vec<$t>, Vec<$t>> for Vec<$t> {
            #[inline]
            fn sub(&self, other: &Vec<$t>) -> Vec<$t> {
                let n1 = self.len();
                let n2 = other.len();
                assert!(n1 > 0);
                assert_eq!(n1, n2);
                self.iter().zip(other.iter()).map(|(a, b)| a - b).collect()
            }
        }

        impl ArgminSub<Vec<Vec<$t>>, Vec<Vec<$t>>> for Vec<Vec<$t>> {
            #[inline]
            fn sub(&self, other: &Vec<Vec<$t>>) -> Vec<Vec<$t>> {
                let sr = self.len();
                let or = other.len();
                assert!(sr > 0);
                // implicitly, or > 0
                assert_eq!(sr, or);
                let sc = self[0].len();
                self.iter()
                    .zip(other.iter())
                    .map(|(a, b)| {
                        assert_eq!(a.len(), sc);
                        assert_eq!(b.len(), sc);
                        <Vec<$t> as ArgminSub<Vec<$t>, Vec<$t>>>::sub(&a, &b)
                    })
                    .collect()
            }
        }

        impl ArgminSub<$t, Vec<Vec<$t>>> for Vec<Vec<$t>> {
            #[inline]
            fn sub(&self, other: &$t) -> Vec<Vec<$t>> {
                let sr = self.len();
                assert!(sr > 0);
                let sc = self[0].len();
                self.iter()
                    .map(|a| {
                        assert_eq!(a.len(), sc);
                        <Vec<$t> as ArgminSub<$t, Vec<$t>>>::sub(&a, &other)
                    })
                    .collect()
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
make_sub!(Complex<i8>);
make_sub!(Complex<i16>);
make_sub!(Complex<i32>);
make_sub!(Complex<i64>);
make_sub!(Complex<u8>);
make_sub!(Complex<u16>);
make_sub!(Complex<u32>);
make_sub!(Complex<u64>);
make_sub!(Complex<f32>);
make_sub!(Complex<f64>);

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_sub_vec_scalar_ $t>]() {
                    let a = vec![100 as $t, 40 as $t, 76 as $t];
                    let b = 34 as $t;
                    let target = vec![66 as $t, 6 as $t, 42 as $t];
                    let res = <Vec<$t> as ArgminSub<$t, Vec<$t>>>::sub(&a, &b);
                    for i in 0..3 {
                        assert_relative_eq!(target[i] as f64, res[i] as f64, epsilon = std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_sub_vec_scalar_complex_ $t>]() {
                    let a = vec![
                        Complex::new(100 as $t, 40 as $t),
                        Complex::new(76 as $t, 42 as $t),
                        Complex::new(44 as $t, 35 as $t),
                    ];
                    let b = Complex::new(34 as $t, 12 as $t);
                    let target = vec![a[0] - b, a[1] - b, a[2] - b];
                    let res = <Vec<Complex<$t>> as ArgminSub<Complex<$t>, Vec<Complex<$t>>>>::sub(&a, &b);
                    for i in 0..3 {
                        assert_relative_eq!(target[i].re as f64, res[i].re as f64, epsilon = std::f64::EPSILON);
                        assert_relative_eq!(target[i].im as f64, res[i].im as f64, epsilon = std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_sub_scalar_vec_ $t>]() {
                    let a = vec![1 as $t, 4 as $t, 58 as $t];
                    let b = 100 as $t;
                    let target = vec![99 as $t, 96 as $t, 42 as $t];
                    let res = <$t as ArgminSub<Vec<$t>, Vec<$t>>>::sub(&b, &a);
                    for i in 0..3 {
                        assert_relative_eq!(target[i] as f64, res[i] as f64, epsilon = std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_sub_scalar_vec_complex_ $t>]() {
                    let a = vec![
                        Complex::new(1 as $t, 4 as $t),
                        Complex::new(12 as $t, 21 as $t),
                        Complex::new(6 as $t, 10 as $t)
                    ];
                    let b = Complex::new(100 as $t, 50 as $t);
                    let target = vec![b - a[0], b - a[1], b - a[2]];
                    let res = <Complex<$t> as ArgminSub<Vec<Complex<$t>>, Vec<Complex<$t>>>>::sub(&b, &a);
                    for i in 0..3 {
                        assert_relative_eq!(target[i].re as f64, res[i].re as f64, epsilon = std::f64::EPSILON);
                        assert_relative_eq!(target[i].im as f64, res[i].im as f64, epsilon = std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_sub_vec_vec_ $t>]() {
                    let a = vec![43 as $t, 46 as $t, 50 as $t];
                    let b = vec![1 as $t, 4 as $t, 8 as $t];
                    let target = vec![42 as $t, 42 as $t, 42 as $t];
                    let res = <Vec<$t> as ArgminSub<Vec<$t>, Vec<$t>>>::sub(&a, &b);
                    for i in 0..3 {
                        assert_relative_eq!(target[i] as f64, res[i] as f64, epsilon = std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_sub_vec_vec_complex_ $t>]() {
                    let a = vec![
                        Complex::new(100 as $t, 40 as $t),
                        Complex::new(76 as $t, 42 as $t),
                        Complex::new(44 as $t, 35 as $t),
                    ];
                    let b = vec![
                        Complex::new(1 as $t, 4 as $t),
                        Complex::new(12 as $t, 21 as $t),
                        Complex::new(6 as $t, 10 as $t)
                    ];
                    let target = vec![a[0] - b[0], a[1] - b[1], a[2] - b[2]];
                    let res = <Vec<Complex<$t>> as ArgminSub<Vec<Complex<$t>>, Vec<Complex<$t>>>>::sub(&a, &b);
                    for i in 0..3 {
                        assert_relative_eq!(target[i].re as f64, res[i].re as f64, epsilon = std::f64::EPSILON);
                        assert_relative_eq!(target[i].im as f64, res[i].im as f64, epsilon = std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_sub_vec_vec_panic_ $t>]() {
                    let a = vec![41 as $t, 38 as $t, 34 as $t];
                    let b = vec![1 as $t, 4 as $t];
                    <Vec<$t> as ArgminSub<Vec<$t>, Vec<$t>>>::sub(&a, &b);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_sub_vec_vec_panic_2_ $t>]() {
                    let a = vec![];
                    let b = vec![41 as $t, 38 as $t, 34 as $t];
                    <Vec<$t> as ArgminSub<Vec<$t>, Vec<$t>>>::sub(&a, &b);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_sub_vec_vec_panic_3_ $t>]() {
                    let a = vec![41 as $t, 38 as $t, 34 as $t];
                    let b = vec![];
                    <Vec<$t> as ArgminSub<Vec<$t>, Vec<$t>>>::sub(&a, &b);
                }
            }

            item! {
                #[test]
                fn [<test_sub_mat_mat_ $t>]() {
                    let a = vec![
                        vec![43 as $t, 46 as $t, 50 as $t],
                        vec![44 as $t, 47 as $t, 51 as $t]
                    ];
                    let b = vec![
                        vec![1 as $t, 4 as $t, 8 as $t],
                        vec![2 as $t, 5 as $t, 9 as $t]
                    ];
                    let target = vec![
                        vec![42 as $t, 42 as $t, 42 as $t],
                        vec![42 as $t, 42 as $t, 42 as $t]
                    ];
                    let res = <Vec<Vec<$t>> as ArgminSub<Vec<Vec<$t>>, Vec<Vec<$t>>>>::sub(&a, &b);
                    for i in 0..3 {
                        for j in 0..2 {
                            assert_relative_eq!(target[j][i] as f64, res[j][i] as f64, epsilon = std::f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_sub_mat_mat_complex_ $t>]() {
                    let a = vec![
                        vec![Complex::new(100 as $t, 40 as $t), Complex::new(76 as $t, 42 as $t)],
                        vec![Complex::new(44 as $t, 30 as $t), Complex::new(56 as $t, 52 as $t)],
                        vec![Complex::new(64 as $t, 40 as $t), Complex::new(86 as $t, 72 as $t)],
                    ];
                    let b = vec![
                        vec![Complex::new(10 as $t, 4 as $t), Complex::new(7 as $t, 4 as $t)],
                        vec![Complex::new(4 as $t, 3 as $t), Complex::new(6 as $t, 5 as $t)],
                        vec![Complex::new(6 as $t, 4 as $t), Complex::new(8 as $t, 2 as $t)],
                    ];
                    let target = vec![
                        vec![a[0][0] - b[0][0], a[0][1] - b[0][1]],
                        vec![a[1][0] - b[1][0], a[1][1] - b[1][1]],
                        vec![a[2][0] - b[2][0], a[2][1] - b[2][1]],
                    ];
                    let res = <Vec<Vec<Complex<$t>>> as ArgminSub<Vec<Vec<Complex<$t>>>, Vec<Vec<Complex<$t>>>>>::sub(&a, &b);
                    for i in 0..2 {
                        for j in 0..3 {
                            assert_relative_eq!(target[j][i].re as f64, res[j][i].re as f64, epsilon = std::f64::EPSILON);
                            assert_relative_eq!(target[j][i].im as f64, res[j][i].im as f64, epsilon = std::f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_sub_mat_scalar_ $t>]() {
                    let a = vec![
                        vec![43 as $t, 46 as $t, 50 as $t],
                        vec![44 as $t, 47 as $t, 51 as $t]
                    ];
                    let b = 2 as $t;
                    let target = vec![
                        vec![41 as $t, 44 as $t, 48 as $t],
                        vec![42 as $t, 45 as $t, 49 as $t]
                    ];
                    let res = <Vec<Vec<$t>> as ArgminSub<$t, Vec<Vec<$t>>>>::sub(&a, &b);
                    for i in 0..3 {
                        for j in 0..2 {
                            assert_relative_eq!(target[j][i] as f64, res[j][i] as f64, epsilon = std::f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_sub_mat_scalar_complex_ $t>]() {
                    let a = vec![
                        vec![Complex::new(100 as $t, 40 as $t), Complex::new(76 as $t, 42 as $t)],
                        vec![Complex::new(44 as $t, 30 as $t), Complex::new(56 as $t, 52 as $t)],
                        vec![Complex::new(64 as $t, 40 as $t), Complex::new(86 as $t, 72 as $t)],
                    ];
                    let b = Complex::new(2 as $t, 14 as $t);
                    let target = vec![
                        vec![a[0][0] - b, a[0][1] - b],
                        vec![a[1][0] - b, a[1][1] - b],
                        vec![a[2][0] - b, a[2][1] - b],
                    ];
                    let res = <Vec<Vec<Complex<$t>>> as ArgminSub<Complex<$t>, Vec<Vec<Complex<$t>>>>>::sub(&a, &b);
                    for i in 0..2 {
                        for j in 0..3 {
                            assert_relative_eq!(target[j][i].re as f64, res[j][i].re as f64, epsilon = std::f64::EPSILON);
                            assert_relative_eq!(target[j][i].im as f64, res[j][i].im as f64, epsilon = std::f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_sub_mat_mat_panic_1_ $t>]() {
                    let a = vec![
                        vec![41 as $t, 38 as $t, 34 as $t],
                        vec![40 as $t, 37 as $t, 33 as $t]
                    ];
                    let b = vec![
                        vec![1 as $t, 4 as $t, 8 as $t],
                        vec![2 as $t, 9 as $t]
                    ];
                    <Vec<Vec<$t>> as ArgminSub<Vec<Vec<$t>>, Vec<Vec<$t>>>>::sub(&a, &b);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_sub_mat_mat_panic_2_ $t>]() {
                    let a = vec![
                        vec![41 as $t, 38 as $t, 34 as $t],
                    ];
                    let b = vec![
                        vec![1 as $t, 4 as $t, 8 as $t],
                        vec![2 as $t, 5 as $t, 9 as $t]
                    ];
                    <Vec<Vec<$t>> as ArgminSub<Vec<Vec<$t>>, Vec<Vec<$t>>>>::sub(&a, &b);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_sub_mat_mat_panic_3_ $t>]() {
                    let a = vec![
                        vec![1 as $t, 4 as $t, 8 as $t],
                        vec![2 as $t, 5 as $t, 9 as $t]
                    ];
                    let b = vec![];
                    <Vec<Vec<$t>> as ArgminSub<Vec<Vec<$t>>, Vec<Vec<$t>>>>::sub(&a, &b);
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
