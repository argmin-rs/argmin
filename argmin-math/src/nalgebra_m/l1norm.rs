// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminL1Norm;

use nalgebra::{
    base::{dimension::Dim, storage::Storage},
    LpNorm, Matrix, SimdComplexField,
};

impl<N, R, C, S> ArgminL1Norm<N::SimdRealField> for Matrix<N, R, C, S>
where
    N: SimdComplexField,
    R: Dim,
    C: Dim,
    S: Storage<N, R, C>,
{
    #[inline]
    fn l1_norm(&self) -> N::SimdRealField {
        self.apply_norm(&LpNorm(1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector2;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_l1norm_ $t>]() {
                    let a = Vector2::new(4 as $t, 3 as $t);
                    let res = <Vector2<$t> as ArgminL1Norm<$t>>::l1_norm(&a);
                    let target = 7 as $t;
                    assert!(((target - res) as f64).abs() < std::f64::EPSILON);
                }
            }
        };
    }

    macro_rules! make_test_signed {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_l1norm_signed_ $t>]() {
                    let a = Vector2::new(-4 as $t, -3 as $t);
                    let res = <Vector2<$t> as ArgminL1Norm<$t>>::l1_norm(&a);
                    let target = 7 as $t;
                    assert!(((target - res) as f64).abs() < std::f64::EPSILON);
                }
            }
        };
    }

    make_test!(f32);
    make_test!(f64);

    make_test_signed!(f32);
    make_test_signed!(f64);
}
