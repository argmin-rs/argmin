// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::{Allocator, ArgminAdd, SameShapeAllocator};

use crate::ClosedAdd;
use nalgebra::{
    base::{
        constraint::{SameNumberOfColumns, SameNumberOfRows, ShapeConstraint},
        dimension::Dim,
        storage::Storage,
        Scalar,
    },
    DefaultAllocator, Matrix, MatrixSum, OMatrix,
};

impl<N, R, C, S> ArgminAdd<N, OMatrix<N, R, C>> for Matrix<N, R, C, S>
where
    N: Scalar + ClosedAdd + Copy,
    R: Dim,
    C: Dim,
    S: Storage<N, R, C>,
    DefaultAllocator: Allocator<N, R, C>,
{
    #[inline]
    fn add(&self, other: &N) -> OMatrix<N, R, C> {
        self.add_scalar(*other)
    }
}

impl<N, R, C, S> ArgminAdd<Matrix<N, R, C, S>, OMatrix<N, R, C>> for N
where
    N: Scalar + ClosedAdd + Copy,
    R: Dim,
    C: Dim,
    S: Storage<N, R, C>,
    DefaultAllocator: Allocator<N, R, C>,
{
    #[inline]
    fn add(&self, other: &Matrix<N, R, C, S>) -> OMatrix<N, R, C> {
        other.add_scalar(*self)
    }
}

impl<N, R1, C1, R2, C2, SA, SB> ArgminAdd<Matrix<N, R2, C2, SB>, MatrixSum<N, R1, C1, R2, C2>>
    for Matrix<N, R1, C1, SA>
where
    N: Scalar + ClosedAdd,
    R1: Dim,
    C1: Dim,
    R2: Dim,
    C2: Dim,
    SA: Storage<N, R1, C1>,
    SB: Storage<N, R2, C2>,
    DefaultAllocator: SameShapeAllocator<N, R1, C1, R2, C2>,
    ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2>,
{
    #[inline]
    fn add(&self, other: &Matrix<N, R2, C2, SB>) -> MatrixSum<N, R1, C1, R2, C2> {
        self + other
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::{DMatrix, DVector, Matrix2x3, Vector3};
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_add_vec_scalar_ $t>]() {
                    let a = Vector3::new(1 as $t, 4 as $t, 8 as $t);
                    let b = 34 as $t;
                    let target = Vector3::new(35 as $t, 38 as $t, 42 as $t);
                    let res = <Vector3<$t> as ArgminAdd<$t, Vector3<$t>>>::add(&a, &b);
                    for i in 0..3 {
                        assert_relative_eq!(target[i] as f64, res[i] as f64, epsilon = f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_add_scalar_vec_ $t>]() {
                    let a = Vector3::new(1 as $t, 4 as $t, 8 as $t);
                    let b = 34 as $t;
                    let target = Vector3::new(35 as $t, 38 as $t, 42 as $t);
                    let res = <$t as ArgminAdd<Vector3<$t>, Vector3<$t>>>::add(&b, &a);
                    for i in 0..3 {
                        assert_relative_eq!(target[i] as f64, res[i] as f64, epsilon = f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_add_vec_vec_ $t>]() {
                    let a = Vector3::new(1 as $t, 4 as $t, 8 as $t);
                    let b = Vector3::new(41 as $t, 38 as $t, 34 as $t);
                    let target = Vector3::new(42 as $t, 42 as $t, 42 as $t);
                    let res = <Vector3<$t> as ArgminAdd<Vector3<$t>, Vector3<$t>>>::add(&a, &b);
                    for i in 0..3 {
                        assert_relative_eq!(target[i] as f64, res[i] as f64, epsilon = f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_add_vec_vec_panic_ $t>]() {
                    let a = DVector::from_vec(vec![1 as $t, 4 as $t]);
                    let b = DVector::from_vec(vec![41 as $t, 38 as $t, 34 as $t]);
                    <DVector<$t> as ArgminAdd<DVector<$t>, DVector<$t>>>::add(&a, &b);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_add_vec_vec_panic_2_ $t>]() {
                    let a = DVector::from_vec(vec![]);
                    let b = DVector::from_vec(vec![41 as $t, 38 as $t, 34 as $t]);
                    <DVector<$t> as ArgminAdd<DVector<$t>, DVector<$t>>>::add(&a, &b);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_add_vec_vec_panic_3_ $t>]() {
                    let a = DVector::from_vec(vec![41 as $t, 38 as $t, 34 as $t]);
                    let b = DVector::from_vec(vec![]);
                    <DVector<$t> as ArgminAdd<DVector<$t>, DVector<$t>>>::add(&a, &b);
                }
            }

            item! {
                #[test]
                fn [<test_add_mat_mat_ $t>]() {
                    let a = Matrix2x3::new(
                        1 as $t, 4 as $t, 8 as $t,
                        2 as $t, 5 as $t, 9 as $t
                    );
                    let b = Matrix2x3::new(
                        41 as $t, 38 as $t, 34 as $t,
                        40 as $t, 37 as $t, 33 as $t
                    );
                    let target = Matrix2x3::new(
                        42 as $t, 42 as $t, 42 as $t,
                        42 as $t, 42 as $t, 42 as $t
                    );
                    let res = <Matrix2x3<$t> as ArgminAdd<Matrix2x3<$t>, Matrix2x3<$t>>>::add(&a, &b);
                    for i in 0..3 {
                        for j in 0..2 {
                            assert_relative_eq!(target[(j, i)] as f64, res[(j, i)] as f64, epsilon = f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_add_mat_scalar_ $t>]() {
                    let a = Matrix2x3::new(
                        1 as $t, 4 as $t, 8 as $t,
                        2 as $t, 5 as $t, 9 as $t
                    );
                    let b = 2 as $t;
                    let target = Matrix2x3::new(
                        3 as $t, 6 as $t, 10 as $t,
                        4 as $t, 7 as $t, 11 as $t
                    );
                    let res = <Matrix2x3<$t> as ArgminAdd<$t, Matrix2x3<$t>>>::add(&a, &b);
                    for i in 0..3 {
                        for j in 0..2 {
                            assert_relative_eq!(target[(j, i)] as f64, res[(j, i)] as f64, epsilon = f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_add_mat_mat_panic_2_ $t>]() {
                    let a = DMatrix::from_vec(2, 3, vec![
                        1 as $t, 4 as $t, 8 as $t,
                        2 as $t, 5 as $t, 9 as $t
                    ]);
                    let b = DMatrix::from_vec(1, 2, vec![
                        41 as $t, 38 as $t,
                    ]);
                    <DMatrix<$t> as ArgminAdd<DMatrix<$t>, DMatrix<$t>>>::add(&a, &b);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_add_mat_mat_panic_3_ $t>]() {
                    let a = DMatrix::from_vec(2, 3, vec![
                        1 as $t, 4 as $t, 8 as $t,
                        2 as $t, 5 as $t, 9 as $t
                    ]);
                    let b = DMatrix::from_vec(0, 0, vec![]);
                    <DMatrix<$t> as ArgminAdd<DMatrix<$t>, DMatrix<$t>>>::add(&a, &b);
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
