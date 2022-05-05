// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::{ArgminDot, ArgminTDot};

use num_traits::{One, Zero};

use nalgebra::{
    base::{
        allocator::Allocator,
        constraint::{AreMultipliable, DimEq, ShapeConstraint},
        dimension::Dim,
        storage::Storage,
        Scalar,
    },
    ClosedAdd, ClosedMul, DefaultAllocator, Matrix, OMatrix,
};

impl<N, R1, R2, C1, C2, SA, SB> ArgminDot<Matrix<N, R2, C2, SB>, N> for Matrix<N, R1, C1, SA>
where
    N: Scalar + Zero + ClosedAdd + ClosedMul,
    R1: Dim,
    R2: Dim,
    C1: Dim,
    C2: Dim,
    SA: Storage<N, R1, C1>,
    SB: Storage<N, R2, C2>,
    ShapeConstraint: DimEq<R1, R2> + DimEq<C1, C2>,
{
    #[inline]
    #[allow(clippy::only_used_in_recursion)]
    fn dot(&self, other: &Matrix<N, R2, C2, SB>) -> N {
        self.dot(other)
    }
}

impl<N, R, C, S> ArgminDot<N, OMatrix<N, R, C>> for Matrix<N, R, C, S>
where
    N: Scalar + Copy + ClosedMul,
    R: Dim,
    C: Dim,
    S: Storage<N, R, C>,
    DefaultAllocator: Allocator<N, R, C>,
{
    #[inline]
    fn dot(&self, other: &N) -> OMatrix<N, R, C> {
        self * *other
    }
}

impl<N, R, C, S> ArgminDot<Matrix<N, R, C, S>, OMatrix<N, R, C>> for N
where
    N: Scalar + Copy + ClosedMul,
    R: Dim,
    C: Dim,
    S: Storage<N, R, C>,
    DefaultAllocator: Allocator<N, R, C>,
{
    #[inline]
    fn dot(&self, other: &Matrix<N, R, C, S>) -> OMatrix<N, R, C> {
        other * *self
    }
}

impl<N, R1, R2, C1, C2, SA, SB> ArgminDot<Matrix<N, R2, C2, SB>, OMatrix<N, R1, C2>>
    for Matrix<N, R1, C1, SA>
where
    N: Scalar + Zero + One + ClosedAdd + ClosedMul,
    R1: Dim,
    R2: Dim,
    C1: Dim,
    C2: Dim,
    SA: Storage<N, R1, C1>,
    SB: Storage<N, R2, C2>,
    DefaultAllocator: Allocator<N, R1, C2>,
    ShapeConstraint: AreMultipliable<R1, C1, R2, C2>,
{
    #[inline]
    fn dot(&self, other: &Matrix<N, R2, C2, SB>) -> OMatrix<N, R1, C2> {
        self * other
    }
}

impl<N, R1, R2, C1, C2, SA, SB> ArgminTDot<Matrix<N, R2, C2, SB>, OMatrix<N, C2, C1>>
    for Matrix<N, R1, C1, SA>
where
    N: Scalar + Zero + One + ClosedAdd + ClosedMul,
    R1: Dim,
    R2: Dim,
    C1: Dim,
    C2: Dim,
    SA: Storage<N, R1, C1>,
    SB: Storage<N, R2, C2>,
    DefaultAllocator: Allocator<N, R1, C2>,
    DefaultAllocator: Allocator<N, C1, R1>,
    DefaultAllocator: Allocator<N, C1, C2>,
    DefaultAllocator: Allocator<N, C2, C1>,
    ShapeConstraint: AreMultipliable<C1, R1, R2, C2>,
{
    #[inline]
    fn tdot(&self, other: &Matrix<N, R2, C2, SB>) -> OMatrix<N, C2, C1> {
        (self.transpose() * other).transpose()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{
        DMatrix, Matrix2x3, Matrix3, RowDVector, RowVector2, RowVector3, Vector2, Vector3,
    };
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_vec_vec_ $t>]() {
                    let a = Vector3::new(1 as $t, 2 as $t, 3 as $t);
                    let b = Vector3::new(4 as $t, 5 as $t, 6 as $t);
                    let res: $t = <Vector3<$t> as ArgminDot<Vector3<$t>, $t>>::dot(&a, &b);
                    assert!((((res - 32 as $t) as f64).abs()) < std::f64::EPSILON);
                }
            }

            item! {
                #[test]
                fn [<test_vec_scalar_ $t>]() {
                    let a = Vector3::new(1 as $t, 2 as $t, 3 as $t);
                    let b = 2 as $t;
                    let product: Vector3<$t> =
                        <Vector3<$t> as ArgminDot<$t, Vector3<$t>>>::dot(&a, &b);
                    let res = Vector3::new(2 as $t, 4 as $t, 6 as $t);
                    for i in 0..3 {
                        assert!((((res[i] - product[i]) as f64).abs()) < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_scalar_vec_ $t>]() {
                    let a = Vector3::new(1 as $t, 2 as $t, 3 as $t);
                    let b = 2 as $t;
                    let product: Vector3<$t> =
                        <$t as ArgminDot<Vector3<$t>, Vector3<$t>>>::dot(&b, &a);
                    let res = Vector3::new(2 as $t, 4 as $t, 6 as $t);
                    for i in 0..3 {
                        assert!((((res[i] - product[i]) as f64).abs()) < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mat_vec_ $t>]() {
                    let a = Vector3::new(1 as $t, 2 as $t, 3 as $t);
                    let b = RowVector3::new(4 as $t, 5 as $t, 6 as $t);
                    let res = Matrix3::new(
                        4 as $t, 5 as $t, 6 as $t,
                        8 as $t, 10 as $t, 12 as $t,
                        12 as $t, 15 as $t, 18 as $t
                    );
                    let product: Matrix3<$t> =
                        <Vector3<$t> as ArgminDot<RowVector3<$t>, Matrix3<$t>>>::dot(&a, &b);
                    for i in 0..3 {
                        for j in 0..3 {
                            assert!((((res[(i, j)] - product[(i, j)]) as f64).abs()) < std::f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mat_vec_2_ $t>]() {
                    let a = Matrix3::new(
                        1 as $t, 2 as $t, 3 as $t,
                        4 as $t, 5 as $t, 6 as $t,
                        7 as $t, 8 as $t, 9 as $t
                    );
                    let b = Vector3::new(1 as $t, 2 as $t, 3 as $t);
                    let res = Vector3::new(14 as $t, 32 as $t, 50 as $t);
                    let product: Vector3<$t> =
                        <Matrix3<$t> as ArgminDot<Vector3<$t>, Vector3<$t>>>::dot(&a, &b);
                    for i in 0..3 {
                        assert!((((res[i] - product[i]) as f64).abs()) < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mat_mat_ $t>]() {
                    let a = Matrix3::new(
                        1 as $t, 2 as $t, 3 as $t,
                        4 as $t, 5 as $t, 6 as $t,
                        3 as $t, 2 as $t, 1 as $t
                    );
                    let b = Matrix3::new(
                        3 as $t, 2 as $t, 1 as $t,
                        6 as $t, 5 as $t, 4 as $t,
                        2 as $t, 4 as $t, 3 as $t
                    );
                    let res = Matrix3::new(
                        21 as $t, 24 as $t, 18 as $t,
                        54 as $t, 57 as $t, 42 as $t,
                        23 as $t, 20 as $t, 14 as $t
                    );
                    let product: Matrix3<$t> =
                        <Matrix3<$t> as ArgminDot<Matrix3<$t>, Matrix3<$t>>>::dot(&a, &b);
                    for i in 0..3 {
                        for j in 0..3 {
                            assert!((((res[(i, j)] - product[(i, j)]) as f64).abs()) < std::f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mat_primitive_ $t>]() {
                    let a = Matrix3::new(
                        1 as $t, 2 as $t, 3 as $t,
                        4 as $t, 5 as $t, 6 as $t,
                        3 as $t, 2 as $t, 1 as $t
                    );
                    let res = Matrix3::new(
                        2 as $t, 4 as $t, 6 as $t,
                        8 as $t, 10 as $t, 12 as $t,
                        6 as $t, 4 as $t, 2 as $t
                    );
                    let product: Matrix3<$t> =
                        <Matrix3<$t> as ArgminDot<$t, Matrix3<$t>>>::dot(&a, &(2 as $t));
                    for i in 0..3 {
                        for j in 0..3 {
                            assert!((((res[(i, j)] - product[(i, j)]) as f64).abs()) < std::f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_primitive_mat_ $t>]() {
                    let a = Matrix3::new(
                        1 as $t, 2 as $t, 3 as $t,
                        4 as $t, 5 as $t, 6 as $t,
                        3 as $t, 2 as $t, 1 as $t
                    );
                    let res = Matrix3::new(
                        2 as $t, 4 as $t, 6 as $t,
                        8 as $t, 10 as $t, 12 as $t,
                        6 as $t, 4 as $t, 2 as $t
                    );
                    let product: Matrix3<$t> =
                        <$t as ArgminDot<Matrix3<$t>, Matrix3<$t>>>::dot(&(2 as $t), &a);
                    for i in 0..3 {
                        for j in 0..3 {
                            assert!((((res[(i, j)] - product[(i, j)]) as f64).abs()) < std::f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mat_vec_vec_ $t>]() {
                    let a = Matrix2x3::new(
                        1 as $t, 2 as $t, 3 as $t,
                        4 as $t, 5 as $t, 6 as $t,
                    );
                    let b = RowVector2::new(1 as $t, 2 as $t);
                    let res = RowVector3::new(9 as $t, 12 as $t, 15 as $t);
                    let product = ArgminDot::<Matrix2x3<$t>, RowVector3<$t>>::dot(&b, &a);
                    for i in 0..3 {
                        assert!((((res[i] - product[i]) as f64).abs()) < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mat_vec_vec_dyn_ $t>]() {
                    let a = DMatrix::from_row_slice(2, 3,
                        &[1 as $t, 2 as $t, 3 as $t, 4 as $t, 5 as $t, 6 as $t]
                    );
                    let b = RowVector2::new(1 as $t, 2 as $t);
                    let res = RowVector3::new(9 as $t, 12 as $t, 15 as $t);
                    let product = ArgminDot::<DMatrix<$t>, RowDVector<$t>>::dot(&b, &a);
                    for i in 0..3 {
                        assert!((((res[i] - product[i]) as f64).abs()) < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mat_row_vec_vec_ $t>]() {
                    let a = RowVector3::new(1 as $t, 2 as $t, 3 as $t);
                    let b = RowVector3::new(4 as $t, 5 as $t, 6 as $t);
                    let res: $t = <RowVector3<$t> as ArgminDot<RowVector3<$t>, $t>>::dot(&a, &b);
                    assert!((((res - 32 as $t) as f64).abs()) < std::f64::EPSILON);
                }
            }

            item! {
                #[test]
                fn [<test_tdot_ $t>]() {
                    let a = Matrix2x3::new(
                        1 as $t, 2 as $t, 3 as $t,
                        4 as $t, 5 as $t, 6 as $t,
                    );
                    let b = Vector2::new(1 as $t, 2 as $t);
                    let res = Vector3::new(9 as $t, 12 as $t, 15 as $t);
                    let product = ArgminTDot::<Matrix2x3<$t>, Vector3<$t>>::tdot(&b, &a);
                    for i in 0..3 {
                        assert!((((res[i] - product[i]) as f64).abs()) < std::f64::EPSILON);
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
