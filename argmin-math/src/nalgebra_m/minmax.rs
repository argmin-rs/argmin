// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminMinMax;

use nalgebra::{
    base::{allocator::Allocator, dimension::Dim, Scalar},
    ClosedMul, DefaultAllocator, OMatrix,
};

impl<N, R, C> ArgminMinMax for OMatrix<N, R, C>
where
    N: Scalar + Copy + ClosedMul + PartialOrd,
    R: Dim,
    C: Dim,
    DefaultAllocator: Allocator<N, R, C>,
{
    #[inline]
    fn max(a: &OMatrix<N, R, C>, b: &OMatrix<N, R, C>) -> OMatrix<N, R, C> {
        a.zip_map(b, |aa, bb| if aa > bb { aa } else { bb })
    }

    #[inline]
    fn min(a: &OMatrix<N, R, C>, b: &OMatrix<N, R, C>) -> OMatrix<N, R, C> {
        a.zip_map(b, |aa, bb| if aa < bb { aa } else { bb })
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
                fn [<test_minmax_vec_vec_ $t>]() {
                    let a = Vector3::new(1 as $t, 4 as $t, 8 as $t);
                    let b = Vector3::new(2 as $t, 3 as $t, 4 as $t);
                    let target_max = Vector3::new(2 as $t, 4 as $t, 8 as $t);
                    let target_min = Vector3::new(1 as $t, 3 as $t, 4 as $t);
                    let res_max = <Vector3<$t> as ArgminMinMax>::max(&a, &b);
                    let res_min = <Vector3<$t> as ArgminMinMax>::min(&a, &b);
                    for i in 0..3 {
                        assert!(((target_max[i] - res_max[i]) as f64).abs() < std::f64::EPSILON);
                        assert!(((target_min[i] - res_min[i]) as f64).abs() < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_minmax_mat_mat_ $t>]() {
                    let a = Matrix2x3::new(
                        1 as $t, 4 as $t, 8 as $t,
                        2 as $t, 5 as $t, 9 as $t
                    );
                    let b = Matrix2x3::new(
                        2 as $t, 3 as $t, 4 as $t,
                        3 as $t, 4 as $t, 5 as $t
                    );
                    let target_max = Matrix2x3::new(
                        2 as $t, 4 as $t, 8 as $t,
                        3 as $t, 5 as $t, 9 as $t
                    );
                    let target_min = Matrix2x3::new(
                        1 as $t, 3 as $t, 4 as $t,
                        2 as $t, 4 as $t, 5 as $t
                    );
                    let res_max = <Matrix2x3<$t> as ArgminMinMax>::max(&a, &b);
                    let res_min = <Matrix2x3<$t> as ArgminMinMax>::min(&a, &b);
                    for i in 0..3 {
                        for j in 0..2 {
                        assert!(((target_max[(j, i)] - res_max[(j, i)]) as f64).abs() < std::f64::EPSILON);
                        assert!(((target_min[(j, i)] - res_min[(j, i)]) as f64).abs() < std::f64::EPSILON);
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
