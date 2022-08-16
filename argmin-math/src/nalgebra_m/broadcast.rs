// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminBroadcast;

use nalgebra::{
    base::{allocator::Allocator, dimension::Dim, storage::Storage, Scalar},
    ClosedAdd, ClosedDiv, ClosedMul, ClosedSub, DefaultAllocator, Matrix, OMatrix,
};

impl<N, R1, C1, R2, C2, SV, SM> ArgminBroadcast<Matrix<N, R2, C2, SV>, OMatrix<N, R1, C1>>
    for Matrix<N, R1, C1, SM>
where
    N: Copy + Scalar + ClosedAdd + ClosedMul + ClosedSub + ClosedDiv,
    R1: Dim,
    C1: Dim,
    R2: Dim,
    C2: Dim,
    SM: Storage<N, R1, C1>,
    SV: Storage<N, R2, C2>,
    DefaultAllocator: Allocator<N, R1, C1>,
{
    #[inline]
    fn broadcast_add(&self, other: &Matrix<N, R2, C2, SV>) -> OMatrix<N, R1, C1> {
        let (n, m) = self.shape();
        match other.shape() {
            (1, _) | (_, 1) => OMatrix::<N, R1, C1>::from_iterator_generic(
                R1::from_usize(n),
                C1::from_usize(m),
                (0..m).flat_map(move |i| (0..n).map(move |j| self[(j, i)] + other[i])),
            ),
            _ => panic!(
                "Can't broadcast matrix {}x{} to matrix {}x{}, yet",
                other.nrows(),
                other.ncols(),
                self.nrows(),
                self.ncols()
            ),
        }
    }

    #[inline]
    fn broadcast_sub(&self, other: &Matrix<N, R2, C2, SV>) -> OMatrix<N, R1, C1> {
        let (n, m) = self.shape();
        match other.shape() {
            (1, _) | (_, 1) => OMatrix::<N, R1, C1>::from_iterator_generic(
                R1::from_usize(n),
                C1::from_usize(m),
                (0..m).flat_map(move |i| (0..n).map(move |j| self[(j, i)] - other[i])),
            ),
            _ => panic!(
                "Can't broadcast matrix {}x{} to matrix {}x{}, yet",
                other.nrows(),
                other.ncols(),
                self.nrows(),
                self.ncols()
            ),
        }
    }

    #[inline]
    fn broadcast_mul(&self, other: &Matrix<N, R2, C2, SV>) -> OMatrix<N, R1, C1> {
        let (n, m) = self.shape();
        match other.shape() {
            (1, _) | (_, 1) => OMatrix::<N, R1, C1>::from_iterator_generic(
                R1::from_usize(n),
                C1::from_usize(m),
                (0..m).flat_map(move |i| (0..n).map(move |j| self[(j, i)] * other[i])),
            ),
            _ => panic!(
                "Can't broadcast matrix {}x{} to matrix {}x{}, yet",
                other.nrows(),
                other.ncols(),
                self.nrows(),
                self.ncols()
            ),
        }
    }

    #[inline]
    fn broadcast_div(&self, other: &Matrix<N, R2, C2, SV>) -> OMatrix<N, R1, C1> {
        let (n, m) = self.shape();
        match other.shape() {
            (1, _) | (_, 1) => OMatrix::<N, R1, C1>::from_iterator_generic(
                R1::from_usize(n),
                C1::from_usize(m),
                (0..m).flat_map(move |i| (0..n).map(move |j| self[(j, i)] / other[i])),
            ),
            _ => panic!(
                "Can't broadcast matrix {}x{} to matrix {}x{}, yet",
                other.nrows(),
                other.ncols(),
                self.nrows(),
                self.ncols()
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix2x3, RowVector3, Vector3};
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_broadcast_add_sub_mul_div_row_vec_ $t>]() {
                    let a = Matrix2x3::new(
                        2 as $t, 4 as $t, 6 as $t,
                        1 as $t, 2 as $t, 3 as $t
                    );
                    let b = RowVector3::new(1 as $t, 2 as $t, 3 as $t);
                    let expected_add = Matrix2x3::new(
                        3 as $t, 6 as $t, 9 as $t,
                        2 as $t, 4 as $t, 6 as $t
                    );
                    let expected_sub = Matrix2x3::new(
                        1 as $t, 2 as $t, 3 as $t,
                        0 as $t, 0 as $t, 0 as $t
                    );
                    let expected_mul = Matrix2x3::new(
                        2 as $t, 8 as $t, 18 as $t,
                        1 as $t, 4 as $t, 9 as $t
                    );
                    let expected_div = Matrix2x3::new(
                        2 as $t, 2 as $t, 2 as $t,
                        1 as $t, 1 as $t, 1 as $t
                    );
                    let res_add = a.broadcast_add(&b);
                    let res_sub = a.broadcast_sub(&b);
                    let res_mul = a.broadcast_mul(&b);
                    let res_div = a.broadcast_div(&b);
                    for i in 0..2 {
                        for j in 0..3 {
                            assert!((((res_add[(i, j)] - expected_add[(i, j)]) as f64).abs()) < std::f64::EPSILON);
                            assert!((((res_sub[(i, j)] - expected_sub[(i, j)]) as f64).abs()) < std::f64::EPSILON);
                            assert!((((res_mul[(i, j)] - expected_mul[(i, j)]) as f64).abs()) < std::f64::EPSILON);
                            assert!((((res_div[(i, j)] - expected_div[(i, j)]) as f64).abs()) < std::f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_broadcast_add_sub_mul_div_vec_ $t>]() {
                    let a = Matrix2x3::new(
                        2 as $t, 4 as $t, 6 as $t,
                        1 as $t, 2 as $t, 3 as $t
                    );
                    let b = Vector3::new(1 as $t, 2 as $t, 3 as $t);
                    let expected_add = Matrix2x3::new(
                        3 as $t, 6 as $t, 9 as $t,
                        2 as $t, 4 as $t, 6 as $t
                    );
                    let expected_sub = Matrix2x3::new(
                        1 as $t, 2 as $t, 3 as $t,
                        0 as $t, 0 as $t, 0 as $t
                    );
                    let expected_mul = Matrix2x3::new(
                        2 as $t, 8 as $t, 18 as $t,
                        1 as $t, 4 as $t, 9 as $t
                    );
                    let expected_div = Matrix2x3::new(
                        2 as $t, 2 as $t, 2 as $t,
                        1 as $t, 1 as $t, 1 as $t
                    );
                    let res_add = a.broadcast_add(&b);
                    let res_sub = a.broadcast_sub(&b);
                    let res_mul = a.broadcast_mul(&b);
                    let res_div = a.broadcast_div(&b);
                    for i in 0..2 {
                        for j in 0..3 {
                            assert!((((res_add[(i, j)] - expected_add[(i, j)]) as f64).abs()) < std::f64::EPSILON);
                            assert!((((res_sub[(i, j)] - expected_sub[(i, j)]) as f64).abs()) < std::f64::EPSILON);
                            assert!((((res_mul[(i, j)] - expected_mul[(i, j)]) as f64).abs()) < std::f64::EPSILON);
                            assert!((((res_div[(i, j)] - expected_div[(i, j)]) as f64).abs()) < std::f64::EPSILON);
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
