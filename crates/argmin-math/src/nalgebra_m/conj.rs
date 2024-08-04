// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminConj;

use nalgebra::{
    base::{allocator::Allocator, dimension::Dim},
    DefaultAllocator, OMatrix, SimdComplexField,
};

impl<N, R, C> ArgminConj for OMatrix<N, R, C>
where
    N: SimdComplexField,
    R: Dim,
    C: Dim,
    DefaultAllocator: Allocator<N, R, C>,
{
    #[inline]
    fn conj(&self) -> OMatrix<N, R, C> {
        self.conjugate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::Vector3;
    use num_complex::Complex;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_conj_complex_nalgebra_ $t>]() {
                    let a = Vector3::new(
                        Complex::new(1 as $t, 2 as $t),
                        Complex::new(4 as $t, -3 as $t),
                        Complex::new(8 as $t, 0 as $t)
                    );
                    let b = Vector3::new(
                        Complex::new(1 as $t, -2 as $t),
                        Complex::new(4 as $t, 3 as $t),
                        Complex::new(8 as $t, 0 as $t)
                    );
                    let res = <Vector3<Complex<$t>> as ArgminConj>::conj(&a);
                    for i in 0..3 {
                        assert_relative_eq!(b[i].re, res[i].re, epsilon = $t::EPSILON);
                        assert_relative_eq!(b[i].im, res[i].im, epsilon = $t::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_conj_nalgebra_ $t>]() {
                    let a = Vector3::new(1 as $t, 4 as $t, 8 as $t);
                    let b = Vector3::new(1 as $t, 4 as $t, 8 as $t);
                    let res = <Vector3<$t> as ArgminConj>::conj(&a);
                    for i in 0..3 {
                        assert_relative_eq!(b[i], res[i], epsilon = $t::EPSILON);
                    }
                }
            }
        };
    }

    make_test!(f32);
    make_test!(f64);
}
