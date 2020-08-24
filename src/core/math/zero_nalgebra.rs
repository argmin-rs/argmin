// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::math::{ArgminZero, ArgminZeroLike};

use num::Zero;

use nalgebra::{
    base::{allocator::Allocator, dimension::Dim},
    DefaultAllocator, MatrixMN, Scalar,
};

impl<N, R, C> ArgminZeroLike for MatrixMN<N, R, C>
where
    N: Scalar + Zero + ArgminZero,
    R: Dim,
    C: Dim,
    DefaultAllocator: Allocator<N, R, C>,
{
    #[inline]
    fn zero_like(&self) -> MatrixMN<N, R, C> {
        Self::zeros_generic(R::from_usize(self.nrows()), C::from_usize(self.ncols()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DVector, Matrix2, Vector2, Vector4};
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_zero_like_ $t>]() {
                    let t: DVector<$t> = DVector::from_vec(vec![]);
                    let a = t.zero_like();
                    assert_eq!(t, a);
                }
            }

            item! {
                #[test]
                fn [<test_zero_like_2_ $t>]() {
                    let a = Vector4::new(42 as $t, 42 as $t, 42 as $t, 42 as $t).zero_like();
                    for i in 0..4 {
                        assert!(((0 as $t - a[i]) as f64).abs() < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_2d_zero_like_ $t>]() {
                    let t: Vector2<$t> = Vector2::zeros();
                    let a = t.zero_like();
                    assert_eq!(t, a);
                }
            }

            item! {
                #[test]
                fn [<test_2d_zero_like_2_ $t>]() {
                    let a = Matrix2::new(
                      42 as $t, 42 as $t,
                      42 as $t, 42 as $t
                    ).zero_like();

                    for i in 0..2 {
                        for j in 0..2 {
                            assert!(((0 as $t - a[(i, j)]) as f64).abs() < std::f64::EPSILON);
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
