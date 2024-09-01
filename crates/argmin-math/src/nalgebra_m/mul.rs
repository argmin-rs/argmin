// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::{Allocator, ArgminMul, SameShapeAllocator};

use crate::ClosedMul;
use nalgebra::{
    base::{
        constraint::{SameNumberOfColumns, SameNumberOfRows, ShapeConstraint},
        dimension::Dim,
        storage::Storage,
        MatrixSum, Scalar,
    },
    DefaultAllocator, Matrix, OMatrix,
};

impl<N, R, C, S> ArgminMul<N, OMatrix<N, R, C>> for Matrix<N, R, C, S>
where
    N: Scalar + Copy + ClosedMul,
    R: Dim,
    C: Dim,
    S: Storage<N, R, C>,
    DefaultAllocator: Allocator<N, R, C>,
{
    #[inline]
    fn mul(&self, other: &N) -> OMatrix<N, R, C> {
        self * *other
    }
}

impl<N, R, C, S> ArgminMul<Matrix<N, R, C, S>, OMatrix<N, R, C>> for N
where
    N: Scalar + Copy + ClosedMul,
    R: Dim,
    C: Dim,
    S: Storage<N, R, C>,
    DefaultAllocator: Allocator<N, R, C>,
{
    #[inline]
    fn mul(&self, other: &Matrix<N, R, C, S>) -> OMatrix<N, R, C> {
        other * *self
    }
}

impl<N, R1, R2, C1, C2, SA, SB> ArgminMul<Matrix<N, R2, C2, SB>, MatrixSum<N, R1, C1, R2, C2>>
    for Matrix<N, R1, C1, SA>
where
    N: Scalar + ClosedMul,
    R1: Dim,
    R2: Dim,
    C1: Dim,
    C2: Dim,
    SA: Storage<N, R1, C1>,
    SB: Storage<N, R2, C2>,
    DefaultAllocator: SameShapeAllocator<N, R1, C1, R2, C2>,
    ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2>,
{
    #[inline]
    fn mul(&self, other: &Matrix<N, R2, C2, SB>) -> MatrixSum<N, R1, C1, R2, C2> {
        self.component_mul(other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::{Matrix2x3, Vector3};
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_mul_vec_scalar_ $t>]() {
                    let a = Vector3::new(1 as $t, 4 as $t, 8 as $t);
                    let b = 2 as $t;
                    let target = Vector3::new(2 as $t, 8 as $t, 16 as $t);
                    let res = <Vector3<$t> as ArgminMul<$t, Vector3<$t>>>::mul(&a, &b);
                    for i in 0..3 {
                        assert_relative_eq!(target[i] as f64, res[i] as f64, epsilon = f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mul_scalar_vec_ $t>]() {
                    let a = Vector3::new(1 as $t, 4 as $t, 8 as $t);
                    let b = 2 as $t;
                    let target = Vector3::new(2 as $t, 8 as $t, 16 as $t);
                    let res = <$t as ArgminMul<Vector3<$t>, Vector3<$t>>>::mul(&b, &a);
                    for i in 0..3 {
                        assert_relative_eq!(target[i] as f64, res[i] as f64, epsilon = f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mul_vec_vec_ $t>]() {
                    let a = Vector3::new(1 as $t, 4 as $t, 8 as $t);
                    let b = Vector3::new(2 as $t, 3 as $t, 4 as $t);
                    let target = Vector3::new(2 as $t, 12 as $t, 32 as $t);
                    let res = <Vector3<$t> as ArgminMul<Vector3<$t>, Vector3<$t>>>::mul(&a, &b);
                    for i in 0..3 {
                        assert_relative_eq!(target[i] as f64, res[i] as f64, epsilon = f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mul_mat_mat_ $t>]() {
                    let a = Matrix2x3::new(
                        1 as $t, 4 as $t, 8 as $t,
                        2 as $t, 5 as $t, 9 as $t
                    );
                    let b = Matrix2x3::new(
                        2 as $t, 3 as $t, 4 as $t,
                        3 as $t, 4 as $t, 5 as $t
                    );
                    let target = Matrix2x3::new(
                        2 as $t, 12 as $t, 32 as $t,
                        6 as $t, 20 as $t, 45 as $t
                    );
                    let res = <Matrix2x3<$t> as ArgminMul<Matrix2x3<$t>, Matrix2x3<$t>>>::mul(&a, &b);
                    for i in 0..3 {
                        for j in 0..2 {
                            assert_relative_eq!(target[(j, i)] as f64, res[(j, i)] as f64, epsilon = f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mul_scalar_mat_1_ $t>]() {
                    let a = Matrix2x3::new(
                        1 as $t, 4 as $t, 8 as $t,
                        2 as $t, 5 as $t, 9 as $t
                    );
                    let b = 2 as $t;
                    let target = Matrix2x3::new(
                        2 as $t, 8 as $t, 16 as $t,
                        4 as $t, 10 as $t, 18 as $t
                    );
                    let res = <Matrix2x3<$t> as ArgminMul<$t, Matrix2x3<$t>>>::mul(&a, &b);
                    for i in 0..3 {
                        for j in 0..2 {
                            assert_relative_eq!(target[(j, i)] as f64, res[(j, i)] as f64, epsilon = f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mul_scalar_mat_2_ $t>]() {
                    let b = Matrix2x3::new(
                        1 as $t, 4 as $t, 8 as $t,
                        2 as $t, 5 as $t, 9 as $t
                    );
                    let a = 2 as $t;
                    let target = Matrix2x3::new(
                        2 as $t, 8 as $t, 16 as $t,
                        4 as $t, 10 as $t, 18 as $t
                    );
                    let res = <$t as ArgminMul<Matrix2x3<$t>, Matrix2x3<$t>>>::mul(&a, &b);
                    for i in 0..3 {
                        for j in 0..2 {
                            assert_relative_eq!(target[(j, i)] as f64, res[(j, i)] as f64, epsilon = f64::EPSILON);
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
