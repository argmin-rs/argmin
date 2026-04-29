// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminLInfNorm;

use nalgebra::{
    base::{dimension::Dim, storage::Storage},
    Matrix, SimdComplexField, UniformNorm,
};

impl<N, R, C, S> ArgminLInfNorm<N::SimdRealField> for Matrix<N, R, C, S>
where
    N: SimdComplexField,
    R: Dim,
    C: Dim,
    S: Storage<N, R, C>,
{
    #[inline]
    fn linf_norm(&self) -> N::SimdRealField {
        self.apply_norm(&UniformNorm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::Vector3;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_linfnorm_ $t>]() {
                    let a = Vector3::new(3 as $t, 4 as $t, 1 as $t);
                    let res = <Vector3<$t> as ArgminLInfNorm<$t>>::linf_norm(&a);
                    let target = 4 as $t;
                    assert_relative_eq!(target as $t, res as $t, epsilon = $t::EPSILON);
                }
            }
        };
    }

    macro_rules! make_test_signed {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_linfnorm_signed_ $t>]() {
                    let a = Vector3::new(-4 as $t, -3 as $t, 2 as $t);
                    let res = <Vector3<$t> as ArgminLInfNorm<$t>>::linf_norm(&a);
                    let target = 4 as $t;
                    assert_relative_eq!(target as $t, res as $t, epsilon = $t::EPSILON);
                }
            }
        };
    }

    make_test!(f32);
    make_test!(f64);

    make_test_signed!(f32);
    make_test_signed!(f64);
}
