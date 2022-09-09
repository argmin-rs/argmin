// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminSignum;

use nalgebra::{
    base::{allocator::Allocator, dimension::Dim},
    DefaultAllocator, OMatrix, SimdComplexField,
};

impl<N, R, C> ArgminSignum for OMatrix<N, R, C>
where
    N: SimdComplexField,
    R: Dim,
    C: Dim,
    DefaultAllocator: Allocator<N, R, C>,
{
    #[inline]
    fn signum(self) -> OMatrix<N, R, C> {
        self.map(|v| v.simd_signum())
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
                fn [<test_signum_ $t>]() {
                    let a = Vector3::new(3 as $t, -4 as $t, -8 as $t);
                    let b = Vector3::new(1 as $t, -1 as $t, -1 as $t);
                    let res = <Vector3<$t> as ArgminSignum>::signum(a);
                    for i in 0..3 {
                        let diff = (b[i] as f64 - res[i] as f64).abs();
                        assert!(diff  < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_signum_scalar_mat_2_ $t>]() {
                    let b = Matrix2x3::new(
                        3 as $t, -4 as $t, 8 as $t,
                        -2 as $t, -5 as $t, 9 as $t
                    );
                    let target = Matrix2x3::new(
                        1 as $t, -1 as $t, 1 as $t,
                        -1 as $t, -1 as $t, 1 as $t
                    );
                    let res = b.signum();
                    for i in 0..3 {
                        for j in 0..2 {
                            assert!(((target[(j, i)] - res[(j, i)]) as f64).abs() < std::f64::EPSILON);
                        }
                    }
                }
            }
        };
    }

    make_test!(f32);
    make_test!(f64);
}
