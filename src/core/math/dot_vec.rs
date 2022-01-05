// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::math::ArgminDot;
use crate::core::math::ArgminTranspose;
use num_complex::Complex;

macro_rules! make_dot_vec {
    ($t:ty) => {
        impl<'a> ArgminDot<Vec<$t>, $t> for Vec<$t> {
            #[inline]
            fn dot(&self, other: &Vec<$t>) -> $t {
                self.iter().zip(other.iter()).map(|(a, b)| a * b).sum()
            }
        }

        impl<'a> ArgminDot<$t, Vec<$t>> for Vec<$t> {
            #[inline]
            fn dot(&self, other: &$t) -> Vec<$t> {
                self.iter().map(|a| a * other).collect()
            }
        }

        impl<'a> ArgminDot<Vec<$t>, Vec<$t>> for $t {
            #[inline]
            fn dot(&self, other: &Vec<$t>) -> Vec<$t> {
                other.iter().map(|a| a * self).collect()
            }
        }

        impl ArgminDot<Vec<$t>, Vec<Vec<$t>>> for Vec<$t> {
            #[inline]
            fn dot(&self, other: &Vec<$t>) -> Vec<Vec<$t>> {
                self.iter()
                    .map(|b| other.iter().map(|a| a * b).collect())
                    .collect()
            }
        }

        impl ArgminDot<Vec<$t>, Vec<$t>> for Vec<Vec<$t>> {
            #[inline]
            fn dot(&self, other: &Vec<$t>) -> Vec<$t> {
                (0..self.len()).map(|i| self[i].dot(other)).collect()
            }
        }

        impl ArgminDot<Vec<Vec<$t>>, Vec<Vec<$t>>> for Vec<Vec<$t>> {
            #[inline]
            fn dot(&self, other: &Vec<Vec<$t>>) -> Vec<Vec<$t>> {
                // Would be more efficient if this wasn't necessary!
                let other = other.clone().t();
                let sr = self.len();
                assert!(sr > 0);
                let sc = self[0].len();
                assert!(sc > 0);
                let or = other.len();
                assert!(or > 0);
                let oc = other[0].len();
                assert_eq!(sc, or);
                assert!(oc > 0);
                let v = vec![<$t>::default(); oc];
                let mut out = vec![v; sr];
                for i in 0..sr {
                    assert_eq!(self[i].len(), sc);
                    for j in 0..oc {
                        out[i][j] = self[i].dot(&other[j]);
                    }
                }
                out
            }
        }

        impl<'a> ArgminDot<$t, Vec<Vec<$t>>> for Vec<Vec<$t>> {
            #[inline]
            fn dot(&self, other: &$t) -> Vec<Vec<$t>> {
                (0..self.len())
                    .map(|i| self[i].iter().map(|a| a * other).collect())
                    .collect()
            }
        }

        impl<'a> ArgminDot<Vec<Vec<$t>>, Vec<Vec<$t>>> for $t {
            #[inline]
            fn dot(&self, other: &Vec<Vec<$t>>) -> Vec<Vec<$t>> {
                (0..other.len())
                    .map(|i| other[i].iter().map(|a| a * self).collect())
                    .collect()
            }
        }
    };
}

make_dot_vec!(f32);
make_dot_vec!(f64);
make_dot_vec!(i8);
make_dot_vec!(i16);
make_dot_vec!(i32);
make_dot_vec!(i64);
make_dot_vec!(u8);
make_dot_vec!(u16);
make_dot_vec!(u32);
make_dot_vec!(u64);
make_dot_vec!(isize);
make_dot_vec!(usize);
make_dot_vec!(Complex<f32>);
make_dot_vec!(Complex<f64>);
make_dot_vec!(Complex<i8>);
make_dot_vec!(Complex<i16>);
make_dot_vec!(Complex<i32>);
make_dot_vec!(Complex<i64>);
make_dot_vec!(Complex<u8>);
make_dot_vec!(Complex<u16>);
make_dot_vec!(Complex<u32>);
make_dot_vec!(Complex<u64>);
make_dot_vec!(Complex<isize>);
make_dot_vec!(Complex<usize>);

#[cfg(test)]
mod tests {
    use super::*;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_vec_vec_ $t>]() {
                    let a = vec![1 as $t, 2 as $t, 3 as $t];
                    let b = vec![4 as $t, 5 as $t, 6 as $t];
                    let res: $t = a.dot(&b);
                    assert!((((res - 32 as $t) as f64).abs()) < std::f64::EPSILON);
                }
            }

            item! {
                #[test]
                fn [<test_vec_scalar_ $t>]() {
                    let a = vec![1 as $t, 2 as $t, 3 as $t];
                    let b = 2 as $t;
                    let product = a.dot(&b);
                    let res = vec![2 as $t, 4 as $t, 6 as $t];
                    for i in 0..3 {
                        assert!((((res[i] - product[i]) as f64).abs()) < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_scalar_vec_ $t>]() {
                    let a = vec![1 as $t, 2 as $t, 3 as $t];
                    let b = 2 as $t;
                    let product = b.dot(&a);
                    let res = vec![2 as $t, 4 as $t, 6 as $t];
                    for i in 0..3 {
                        assert!((((res[i] - product[i]) as f64).abs()) < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mat_vec_ $t>]() {
                    let a = vec![1 as $t, 2 as $t, 3 as $t];
                    let b = vec![4 as $t, 5 as $t, 6 as $t];
                    let res = vec![
                        vec![4 as $t, 5 as $t, 6 as $t],
                        vec![8 as $t, 10 as $t, 12 as $t],
                        vec![12 as $t, 15 as $t, 18 as $t]
                    ];
                    let product: Vec<Vec<$t>> = a.dot(&b);
                    for i in 0..3 {
                        for j in 0..3 {
                            assert!((((res[i][j] - product[i][j]) as f64).abs()) < std::f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mat_vec_2_ $t>]() {
                    let a = vec![
                        vec![1 as $t, 2 as $t, 3 as $t],
                        vec![4 as $t, 5 as $t, 6 as $t],
                        vec![7 as $t, 8 as $t, 9 as $t]
                    ];
                    let b = vec![1 as $t, 2 as $t, 3 as $t];
                    let res = vec![14 as $t, 32 as $t, 50 as $t];
                    let product = a.dot(&b);
                    for i in 0..3 {
                        assert!((((res[i] - product[i]) as f64).abs()) < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mat_mat_ $t>]() {
                    let a = vec![
                        vec![1 as $t, 2 as $t, 3 as $t],
                        vec![4 as $t, 5 as $t, 6 as $t],
                        vec![3 as $t, 2 as $t, 1 as $t]
                    ];
                    let b = vec![
                        vec![3 as $t, 2 as $t, 1 as $t],
                        vec![6 as $t, 5 as $t, 4 as $t],
                        vec![2 as $t, 4 as $t, 3 as $t]
                    ];
                    let res = vec![
                        vec![21 as $t, 24 as $t, 18 as $t],
                        vec![54 as $t, 57 as $t, 42 as $t],
                        vec![23 as $t, 20 as $t, 14 as $t]
                    ];
                    let product = a.dot(&b);
                    for i in 0..3 {
                        for j in 0..3 {
                            assert!((((res[i][j] - product[i][j]) as f64).abs()) < std::f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_mat_mat_panic_1_ $t>]() {
                    let a = vec![];
                    let b = vec![
                        vec![3 as $t, 2 as $t, 1 as $t],
                        vec![6 as $t, 5 as $t, 4 as $t],
                        vec![2 as $t, 4 as $t, 3 as $t]
                    ];
                    a.dot(&b);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_mat_mat_panic_2_ $t>]() {
                    let a: Vec<Vec<$t>> = vec![];
                    let b = vec![
                        vec![3 as $t, 2 as $t, 1 as $t],
                        vec![6 as $t, 5 as $t, 4 as $t],
                        vec![2 as $t, 4 as $t, 3 as $t]
                    ];
                    b.dot(&a);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_mat_mat_panic_3_ $t>]() {
                    let a = vec![
                        vec![1 as $t, 2 as $t],
                        vec![4 as $t, 5 as $t],
                        vec![3 as $t, 2 as $t]
                    ];
                    let b = vec![
                        vec![3 as $t, 2 as $t, 1 as $t],
                        vec![6 as $t, 5 as $t, 4 as $t],
                        vec![2 as $t, 4 as $t, 3 as $t]
                    ];
                    a.dot(&b);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_mat_mat_panic_4_ $t>]() {
                    let a = vec![
                        vec![1 as $t, 2 as $t, 3 as $t],
                        vec![4 as $t, 5 as $t, 6 as $t],
                        vec![3 as $t, 2 as $t, 1 as $t]
                    ];
                    let b = vec![
                        vec![3 as $t, 2 as $t],
                        vec![6 as $t, 5 as $t],
                        vec![3 as $t, 2 as $t]
                    ];
                    a.dot(&b);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_mat_mat_panic_5_ $t>]() {
                    let a = vec![
                        vec![1 as $t, 2 as $t, 3 as $t],
                        vec![4 as $t, 5 as $t, 6 as $t],
                        vec![3 as $t, 2 as $t, 1 as $t]
                    ];
                    let b = vec![
                        vec![3 as $t, 2 as $t, 1 as $t],
                        vec![6 as $t, 5 as $t, 4 as $t],
                        vec![2 as $t, 3 as $t]
                    ];
                    a.dot(&b);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_mat_mat_panic_6_ $t>]() {
                    let a = vec![
                        vec![1 as $t, 2 as $t, 3 as $t],
                        vec![4 as $t, 5 as $t],
                        vec![3 as $t, 2 as $t, 1 as $t]
                    ];
                    let b = vec![
                        vec![3 as $t, 2 as $t, 1 as $t],
                        vec![6 as $t, 5 as $t, 4 as $t],
                        vec![2 as $t, 4 as $t, 3 as $t]
                    ];
                    a.dot(&b);
                }
            }

            item! {
                #[test]
                fn [<test_mat_primitive_ $t>]() {
                    let a = vec![
                        vec![1 as $t, 2 as $t, 3 as $t],
                        vec![4 as $t, 5 as $t, 6 as $t],
                        vec![3 as $t, 2 as $t, 1 as $t]
                    ];
                    let res = vec![
                        vec![2 as $t, 4 as $t, 6 as $t],
                        vec![8 as $t, 10 as $t, 12 as $t],
                        vec![6 as $t, 4 as $t, 2 as $t]
                    ];
                    let product = a.dot(&(2 as $t));
                    for i in 0..3 {
                        for j in 0..3 {
                            assert!((((res[i][j] - product[i][j]) as f64).abs()) < std::f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_primitive_mat_ $t>]() {
                    let a = vec![
                        vec![1 as $t, 2 as $t, 3 as $t],
                        vec![4 as $t, 5 as $t, 6 as $t],
                        vec![3 as $t, 2 as $t, 1 as $t]
                    ];
                    let res = vec![
                        vec![2 as $t, 4 as $t, 6 as $t],
                        vec![8 as $t, 10 as $t, 12 as $t],
                        vec![6 as $t, 4 as $t, 2 as $t]
                    ];
                    let product = (2 as $t).dot(&a);
                    for i in 0..3 {
                        for j in 0..3 {
                            assert!((((res[i][j] - product[i][j]) as f64).abs()) < std::f64::EPSILON);
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
