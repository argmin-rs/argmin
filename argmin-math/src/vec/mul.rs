// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminMul;
use num_complex::Complex;

macro_rules! make_mul {
    ($t:ty) => {
        impl ArgminMul<$t, Vec<$t>> for Vec<$t> {
            #[inline]
            fn mul(&self, other: &$t) -> Vec<$t> {
                self.iter().map(|a| a * other).collect()
            }
        }

        impl ArgminMul<Vec<$t>, Vec<$t>> for $t {
            #[inline]
            fn mul(&self, other: &Vec<$t>) -> Vec<$t> {
                other.iter().map(|a| a * self).collect()
            }
        }

        impl ArgminMul<Vec<$t>, Vec<$t>> for Vec<$t> {
            #[inline]
            fn mul(&self, other: &Vec<$t>) -> Vec<$t> {
                let n1 = self.len();
                let n2 = other.len();
                assert!(n1 > 0);
                assert!(n2 > 0);
                assert_eq!(n1, n2);
                self.iter().zip(other.iter()).map(|(a, b)| a * b).collect()
            }
        }

        impl ArgminMul<Vec<Vec<$t>>, Vec<Vec<$t>>> for Vec<Vec<$t>> {
            #[inline]
            fn mul(&self, other: &Vec<Vec<$t>>) -> Vec<Vec<$t>> {
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
                        <Vec<$t> as ArgminMul<Vec<$t>, Vec<$t>>>::mul(&a, &b)
                    })
                    .collect()
            }
        }

        impl ArgminMul<$t, Vec<Vec<$t>>> for Vec<Vec<$t>> {
            #[inline]
            fn mul(&self, other: &$t) -> Vec<Vec<$t>> {
                self.iter().map(|a| a.mul(other)).collect()
            }
        }

        impl ArgminMul<Vec<Vec<$t>>, Vec<Vec<$t>>> for $t {
            #[inline]
            fn mul(&self, other: &Vec<Vec<$t>>) -> Vec<Vec<$t>> {
                other.iter().map(|a| a.mul(self)).collect()
            }
        }
    };
}

make_mul!(isize);
make_mul!(usize);
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
make_mul!(Complex<isize>);
make_mul!(Complex<usize>);
make_mul!(Complex<i8>);
make_mul!(Complex<u8>);
make_mul!(Complex<i16>);
make_mul!(Complex<u16>);
make_mul!(Complex<i32>);
make_mul!(Complex<u32>);
make_mul!(Complex<i64>);
make_mul!(Complex<u64>);
make_mul!(Complex<f32>);
make_mul!(Complex<f64>);

#[cfg(test)]
mod tests {
    use super::*;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_mul_vec_scalar_ $t>]() {
                    let a = vec![1 as $t, 4 as $t, 8 as $t];
                    let b = 2 as $t;
                    let target = vec![2 as $t, 8 as $t, 16 as $t];
                    let res = <Vec<$t> as ArgminMul<$t, Vec<$t>>>::mul(&a, &b);
                    for i in 0..3 {
                        assert!(((target[i] - res[i]) as f64).abs() < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mul_vec_scalar_complex_ $t>]() {
                    let a = vec![
                        Complex::new(5 as $t, 3 as $t),
                        Complex::new(8 as $t, 2 as $t)
                    ];
                    let b = Complex::new(2 as $t, 3 as $t);
                    let target = vec![a[0] * b, a[1] * b];
                    let res = <Vec<Complex<$t>> as ArgminMul<Complex<$t>, Vec<Complex<$t>>>>::mul(&a, &b);
                    for i in 0..2 {
                        assert!((target[i].re as f64 - res[i].re as f64).abs() < std::f64::EPSILON);
                        assert!((target[i].im as f64 - res[i].im as f64).abs() < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mul_scalar_vec_ $t>]() {
                    let a = vec![1 as $t, 4 as $t, 8 as $t];
                    let b = 2 as $t;
                    let target = vec![2 as $t, 8 as $t, 16 as $t];
                    let res = <$t as ArgminMul<Vec<$t>, Vec<$t>>>::mul(&b, &a);
                    for i in 0..3 {
                        assert!(((target[i] - res[i]) as f64).abs() < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mul_scalar_vec_complex_ $t>]() {
                    let a = vec![
                        Complex::new(5 as $t, 3 as $t),
                        Complex::new(8 as $t, 2 as $t)
                    ];
                    let b = Complex::new(2 as $t, 3 as $t);
                    let target = vec![a[0] * b, a[1] * b];
                    let res = <Complex<$t> as ArgminMul<Vec<Complex<$t>>, Vec<Complex<$t>>>>::mul(&b, &a);
                    for i in 0..2 {
                        assert!((target[i].re as f64 - res[i].re as f64).abs() < std::f64::EPSILON);
                        assert!((target[i].im as f64 - res[i].im as f64).abs() < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mul_vec_vec_ $t>]() {
                    let a = vec![1 as $t, 4 as $t, 8 as $t];
                    let b = vec![2 as $t, 3 as $t, 4 as $t];
                    let target = vec![2 as $t, 12 as $t, 32 as $t];
                    let res = <Vec<$t> as ArgminMul<Vec<$t>, Vec<$t>>>::mul(&a, &b);
                    for i in 0..3 {
                        assert!(((target[i] - res[i]) as f64).abs() < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mul_vec_vec_complex_ $t>]() {
                    let a = vec![
                        Complex::new(5 as $t, 3 as $t),
                        Complex::new(8 as $t, 2 as $t)
                    ];
                    let b = vec![
                        Complex::new(2 as $t, 3 as $t),
                        Complex::new(1 as $t, 2 as $t)
                    ];
                    let target = vec![a[0]*b[0], a[1]*b[1]];
                    let res = <Vec<Complex<$t>> as ArgminMul<Vec<Complex<$t>>, Vec<Complex<$t>>>>::mul(&a, &b);
                    for i in 0..2 {
                        assert!((target[i].re as f64 - res[i].re as f64).abs() < std::f64::EPSILON);
                        assert!((target[i].im as f64 - res[i].im as f64).abs() < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_mul_vec_vec_panic_ $t>]() {
                    let a = vec![1 as $t, 4 as $t];
                    let b = vec![41 as $t, 38 as $t, 34 as $t];
                    <Vec<$t> as ArgminMul<Vec<$t>, Vec<$t>>>::mul(&a, &b);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_mul_vec_vec_panic_2_ $t>]() {
                    let a = vec![];
                    let b = vec![41 as $t, 38 as $t, 34 as $t];
                    <Vec<$t> as ArgminMul<Vec<$t>, Vec<$t>>>::mul(&a, &b);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_mul_vec_vec_panic_3_ $t>]() {
                    let a = vec![41 as $t, 38 as $t, 34 as $t];
                    let b = vec![];
                    <Vec<$t> as ArgminMul<Vec<$t>, Vec<$t>>>::mul(&a, &b);
                }
            }

            item! {
                #[test]
                fn [<test_mul_mat_mat_ $t>]() {
                    let a = vec![
                        vec![1 as $t, 4 as $t, 8 as $t],
                        vec![2 as $t, 5 as $t, 9 as $t]
                    ];
                    let b = vec![
                        vec![2 as $t, 3 as $t, 4 as $t],
                        vec![3 as $t, 4 as $t, 5 as $t]
                    ];
                    let target = vec![
                        vec![2 as $t, 12 as $t, 32 as $t],
                        vec![6 as $t, 20 as $t, 45 as $t]
                    ];
                    let res = <Vec<Vec<$t>> as ArgminMul<Vec<Vec<$t>>, Vec<Vec<$t>>>>::mul(&a, &b);
                    for i in 0..3 {
                        for j in 0..2 {
                        assert!(((target[j][i] - res[j][i]) as f64).abs() < std::f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_mul_mat_mat_panic_1_ $t>]() {
                    let a = vec![
                        vec![1 as $t, 4 as $t, 8 as $t],
                        vec![2 as $t, 9 as $t]
                    ];
                    let b = vec![
                        vec![41 as $t, 38 as $t, 34 as $t],
                        vec![40 as $t, 37 as $t, 33 as $t]
                    ];
                    <Vec<Vec<$t>> as ArgminMul<Vec<Vec<$t>>, Vec<Vec<$t>>>>::mul(&a, &b);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_mul_mat_mat_panic_2_ $t>]() {
                    let a = vec![
                        vec![1 as $t, 4 as $t, 8 as $t],
                        vec![2 as $t, 5 as $t, 9 as $t]
                    ];
                    let b = vec![
                        vec![41 as $t, 38 as $t, 34 as $t],
                    ];
                    <Vec<Vec<$t>> as ArgminMul<Vec<Vec<$t>>, Vec<Vec<$t>>>>::mul(&a, &b);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_mul_mat_mat_panic_3_ $t>]() {
                    let a = vec![
                        vec![1 as $t, 4 as $t, 8 as $t],
                        vec![2 as $t, 5 as $t, 9 as $t]
                    ];
                    let b = vec![];
                    <Vec<Vec<$t>> as ArgminMul<Vec<Vec<$t>>, Vec<Vec<$t>>>>::mul(&a, &b);
                }
            }

            item! {
                #[test]
                fn [<test_mul_scalar_mat_1_ $t>]() {
                    let a = vec![
                        vec![1 as $t, 4 as $t, 8 as $t],
                        vec![2 as $t, 5 as $t, 9 as $t]
                    ];
                    let b = 2 as $t;
                    let target = vec![
                        vec![2 as $t, 8 as $t, 16 as $t],
                        vec![4 as $t, 10 as $t, 18 as $t]
                    ];
                    let res = <Vec<Vec<$t>> as ArgminMul<$t, Vec<Vec<$t>>>>::mul(&a, &b);
                    for i in 0..3 {
                        for j in 0..2 {
                            assert!(((target[j][i] - res[j][i]) as f64).abs() < std::f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mul_scalar_mat_1_complex_ $t>]() {
                    let a = vec![
                        vec![Complex::new(5 as $t, 3 as $t), Complex::new(8 as $t, 2 as $t)],
                        vec![Complex::new(4 as $t, 2 as $t), Complex::new(7 as $t, 1 as $t)],
                        vec![Complex::new(3 as $t, 1 as $t), Complex::new(6 as $t, 2 as $t)],
                    ];
                    let b = Complex::new(3 as $t, 2 as $t);
                    let target = vec![
                        vec![a[0][0] * b, a[0][1] * b],
                        vec![a[1][0] * b, a[1][1] * b],
                        vec![a[2][0] * b, a[2][1] * b],
                    ];
                    let res = <Vec<Vec<Complex<$t>>> as ArgminMul<Complex<$t>, Vec<Vec<Complex<$t>>>>>::mul(&a, &b);
                    for i in 0..2 {
                        for j in 0..3 {
                            assert!((target[j][i].re as f64 - res[j][i].re as f64).abs() < std::f64::EPSILON);
                            assert!((target[j][i].im as f64 - res[j][i].im as f64).abs() < std::f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mul_scalar_mat_2_ $t>]() {
                    let b = vec![
                        vec![1 as $t, 4 as $t, 8 as $t],
                        vec![2 as $t, 5 as $t, 9 as $t]
                    ];
                    let a = 2 as $t;
                    let target = vec![
                        vec![2 as $t, 8 as $t, 16 as $t],
                        vec![4 as $t, 10 as $t, 18 as $t]
                    ];
                    let res = <$t as ArgminMul<Vec<Vec<$t>>, Vec<Vec<$t>>>>::mul(&a, &b);
                    for i in 0..3 {
                        for j in 0..2 {
                            assert!(((target[j][i] - res[j][i]) as f64).abs() < std::f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mul_scalar_mat_2_complex_ $t>]() {
                    let a = vec![
                        vec![Complex::new(5 as $t, 3 as $t), Complex::new(8 as $t, 2 as $t)],
                        vec![Complex::new(4 as $t, 2 as $t), Complex::new(7 as $t, 1 as $t)],
                        vec![Complex::new(3 as $t, 1 as $t), Complex::new(6 as $t, 2 as $t)],
                    ];
                    let b = Complex::new(3 as $t, 2 as $t);
                    let target = vec![
                        vec![a[0][0] * b, a[0][1] * b],
                        vec![a[1][0] * b, a[1][1] * b],
                        vec![a[2][0] * b, a[2][1] * b],
                    ];
                    let res = <Complex<$t> as ArgminMul<Vec<Vec<Complex<$t>>>, Vec<Vec<Complex<$t>>>>>::mul(&b, &a);
                    for i in 0..2 {
                        for j in 0..3 {
                            assert!((target[j][i].re as f64 - res[j][i].re as f64).abs() < std::f64::EPSILON);
                            assert!((target[j][i].im as f64 - res[j][i].im as f64).abs() < std::f64::EPSILON);
                        }
                    }
                }
            }
        };
    }

    make_test!(isize);
    make_test!(usize);
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
