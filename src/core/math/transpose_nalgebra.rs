// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

// Note: This is not really the preferred way I think. Maybe this should also be implemented for
// ArrayViews, which would probably make it more efficient.

use crate::core::math::ArgminTranspose;

use nalgebra::{
    base::{allocator::Allocator, dimension::Dim, storage::Storage, Scalar},
    DefaultAllocator, Matrix, MatrixMN,
};

impl<N, R, C, S> ArgminTranspose<MatrixMN<N, C, R>> for Matrix<N, R, C, S>
where
    N: Scalar,
    R: Dim,
    C: Dim,
    S: Storage<N, R, C>,
    DefaultAllocator: Allocator<N, C, R>,
{
    #[inline]
    fn t(self) -> MatrixMN<N, C, R> {
        self.transpose()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix2, Matrix2x3, Matrix3x2, RowVector2, Vector2};
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_transpose_ $t>]() {
                    let a = Vector2::new(1 as $t, 4 as $t);
                    let target = RowVector2::new(1 as $t, 4 as $t);
                    let res = <Vector2<$t> as ArgminTranspose<RowVector2<$t>>>::t(a);
                    for i in 0..2 {
                        assert!(((target[i] - res[i]) as f64).abs() < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_transpose_2d_1_ $t>]() {
                    let a = Matrix2::new(
                        1 as $t, 4 as $t,
                        8 as $t, 7 as $t
                    );
                    let target = Matrix2::new(
                        1 as $t, 8 as $t,
                        4 as $t, 7 as $t
                    );
                    let res = <Matrix2<$t> as ArgminTranspose<Matrix2<$t>>>::t(a);
                    for i in 0..2 {
                        for j in 0..2 {
                            assert!(((target[(i, j)] - res[(i, j)]) as f64).abs() < std::f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_transpose_2d_2_ $t>]() {
                    let a = Matrix3x2::new(
                        1 as $t, 4 as $t,
                        8 as $t, 7 as $t,
                        3 as $t, 6 as $t
                    );
                    let target = Matrix2x3::new(
                        1 as $t, 8 as $t, 3 as $t,
                        4 as $t, 7 as $t, 6 as $t
                    );
                    let res = <Matrix3x2<$t> as ArgminTranspose<Matrix2x3<$t>>>::t(a);
                    for i in 0..2 {
                        for j in 0..3 {
                            assert!(((target[(i, j)] - res[(i, j)]) as f64).abs() < std::f64::EPSILON);
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
