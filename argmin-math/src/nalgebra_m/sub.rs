// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use std::ops::Sub;

use crate::ArgminSub;

use nalgebra::{
    base::{allocator::Allocator, dimension::Dim, storage::Storage, Scalar},
    ClosedSub, DefaultAllocator, Matrix, OMatrix,
};

impl<N, R, C, S> ArgminSub<N, OMatrix<N, R, C>> for Matrix<N, R, C, S>
where
    N: Scalar + Copy + Sub<Output = N>,
    R: Dim,
    C: Dim,
    S: Storage<N, R, C>,
    DefaultAllocator: Allocator<N, R, C>,
{
    #[inline]
    fn sub(&self, other: &N) -> OMatrix<N, R, C> {
        self.map(|entry| entry - *other)
    }
}

impl<N, R, C, S> ArgminSub<Matrix<N, R, C, S>, OMatrix<N, R, C>> for N
where
    N: Scalar + Copy + Sub<Output = N>,
    R: Dim,
    C: Dim,
    S: Storage<N, R, C>,
    DefaultAllocator: Allocator<N, R, C>,
{
    #[inline]
    fn sub(&self, other: &Matrix<N, R, C, S>) -> OMatrix<N, R, C> {
        other.map(|entry| *self - entry)
    }
}

impl<N, R, C, S> ArgminSub<Matrix<N, R, C, S>, OMatrix<N, R, C>> for Matrix<N, R, C, S>
where
    N: Scalar + ClosedSub,
    R: Dim,
    C: Dim,
    S: Storage<N, R, C>,
    DefaultAllocator: Allocator<N, R, C>,
{
    #[inline]
    fn sub(&self, other: &Matrix<N, R, C, S>) -> OMatrix<N, R, C> {
        self - other
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix2x3, Vector3};
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_sub_vec_scalar_ $t>]() {
                    let a = Vector3::new(36 as $t, 39 as $t, 43 as $t);
                    let b = 1 as $t;
                    let target = Vector3::new(35 as $t, 38 as $t, 42 as $t);
                    let res = <Vector3<$t> as ArgminSub<$t, Vector3<$t>>>::sub(&a, &b);
                    for i in 0..3 {
                        assert!(((target[i] - res[i]) as f64).abs() < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_sub_scalar_vec_ $t>]() {
                    let a = Vector3::new(1 as $t, 4 as $t, 8 as $t);
                    let b = 34 as $t;
                    let target = Vector3::new(33 as $t, 30 as $t, 26 as $t);
                    let res = <$t as ArgminSub<Vector3<$t>, Vector3<$t>>>::sub(&b, &a);
                    for i in 0..3 {
                        assert!(((target[i] - res[i]) as f64).abs() < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_sub_vec_vec_ $t>]() {
                    let a = Vector3::new(41 as $t, 38 as $t, 34 as $t);
                    let b = Vector3::new(1 as $t, 4 as $t, 8 as $t);
                    let target =Vector3::new(40 as $t, 34 as $t, 26 as $t);
                    let res = <Vector3<$t> as ArgminSub<Vector3<$t>, Vector3<$t>>>::sub(&a, &b);
                    for i in 0..3 {
                        assert!(((target[i] - res[i]) as f64).abs() < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_sub_mat_mat_ $t>]() {
                    let a = Matrix2x3::new(
                        43 as $t, 46 as $t, 50 as $t,
                        44 as $t, 47 as $t, 51 as $t
                    );
                    let b = Matrix2x3::new(
                        1 as $t, 4 as $t, 8 as $t,
                        2 as $t, 5 as $t, 9 as $t
                    );
                    let target = Matrix2x3::new(
                        42 as $t, 42 as $t, 42 as $t,
                        42 as $t, 42 as $t, 42 as $t
                    );
                    let res = <Matrix2x3<$t> as ArgminSub<Matrix2x3<$t>, Matrix2x3<$t>>>::sub(&a, &b);
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
                    let a = Matrix2x3::new(
                        43 as $t, 46 as $t, 50 as $t,
                        44 as $t, 47 as $t, 51 as $t
                    );
                    let b = 2 as $t;
                    let target = Matrix2x3::new(
                        41 as $t, 44 as $t, 48 as $t,
                        42 as $t, 45 as $t, 49 as $t
                    );
                    let res = <Matrix2x3<$t> as ArgminSub<$t, Matrix2x3<$t>>>::sub(&a, &b);
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
