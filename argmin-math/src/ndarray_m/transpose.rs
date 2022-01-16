// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

// Note: This is not really the preferred way I think. Maybe this should also be implemented for
// ArrayViews, which would probably make it more efficient.

use crate::ArgminTranspose;
use ndarray::{Array1, Array2};
use num_complex::Complex;

macro_rules! make_add {
    ($t:ty) => {
        impl ArgminTranspose<Array1<$t>> for Array1<$t> {
            #[inline]
            fn t(self) -> Array1<$t> {
                self.reversed_axes()
            }
        }

        impl ArgminTranspose<Array2<$t>> for Array2<$t> {
            #[inline]
            fn t(self) -> Array2<$t> {
                self.reversed_axes()
            }
        }
    };
}

make_add!(i8);
make_add!(i16);
make_add!(i32);
make_add!(i64);
make_add!(u8);
make_add!(u16);
make_add!(u32);
make_add!(u64);
make_add!(f32);
make_add!(f64);
make_add!(Complex<f32>);
make_add!(Complex<f64>);

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                // not sure if this is such a smart test :D
                #[test]
                fn [<test_transpose_ $t>]() {
                    let a = array![1 as $t, 4 as $t];
                    let target = array![1 as $t, 4 as $t];
                    let res = <Array1<$t> as ArgminTranspose<Array1<$t>>>::t(a);
                    for i in 0..2 {
                        assert!(((target[i] - res[i]) as f64).abs() < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_transpose_2d_1_ $t>]() {
                    let a = array![
                        [1 as $t, 4 as $t],
                        [8 as $t, 7 as $t]
                    ];
                    let target = array![
                        [1 as $t, 8 as $t],
                        [4 as $t, 7 as $t]
                    ];
                    let res = <Array2<$t> as ArgminTranspose<Array2<$t>>>::t(a);
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
                    let a = array![
                        [1 as $t, 4 as $t],
                        [8 as $t, 7 as $t],
                        [3 as $t, 6 as $t]
                    ];
                    let target = array![
                        [1 as $t, 8 as $t, 3 as $t],
                        [4 as $t, 7 as $t, 6 as $t]
                    ];
                    let res = <Array2<$t> as ArgminTranspose<Array2<$t>>>::t(a);
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
