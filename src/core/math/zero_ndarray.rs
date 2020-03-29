// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::math::{ArgminZero, ArgminZeroLike};
use num::Zero;

impl<T> ArgminZeroLike for ndarray::Array1<T>
where
    T: Zero + ArgminZero + Clone,
{
    #[inline]
    fn zero_like(&self) -> ndarray::Array1<T> {
        ndarray::Array1::zeros(self.raw_dim())
    }

    // #[inline]
    // fn zero() -> ndarray::Array1<T> {
    //     ndarray::Array1::zeros(0)
    // }
}

impl<T> ArgminZeroLike for ndarray::Array2<T>
where
    T: Zero + ArgminZero + Clone,
{
    #[inline]
    fn zero_like(&self) -> ndarray::Array2<T> {
        ndarray::Array2::zeros(self.raw_dim())
    }

    // #[inline]
    // fn zero() -> ndarray::Array2<T> {
    //     ndarray::Array2::zeros((0, 0))
    // }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, Array2};
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            // item! {
            //     #[test]
            //     fn [<test_zero_ $t>]() {
            //         let a = <Array1<$t> as ArgminZero>::zero();
            //         let b: Array1<$t> = array![];
            //         assert_eq!(a, b);
            //     }
            // }

            item! {
                #[test]
                fn [<test_zero_like_ $t>]() {
                    let t: Array1<$t> = array![];
                    let a = t.zero_like();
                    assert_eq!(t, a);
                }
            }

            item! {
                #[test]
                fn [<test_zero_like_2_ $t>]() {
                    let a = (array![42 as $t, 42 as $t, 42 as $t, 42 as $t]).zero_like();
                    for i in 0..4 {
                        assert!(((0 as $t - a[i]) as f64).abs() < std::f64::EPSILON);
                    }
                }
            }

            // item! {
            //     #[test]
            //     fn [<test_2d_zero_ $t>]() {
            //         let a = <Array2<$t> as ArgminZero>::zero();
            //         let b: Array2<$t> = Array2::zeros((0, 0));
            //         assert_eq!(a, b);
            //     }
            // }

            item! {
                #[test]
                fn [<test_2d_zero_like_ $t>]() {
                    let t: Array2<$t> = Array2::zeros((0, 0));
                    let a = t.zero_like();
                    assert_eq!(t, a);
                }
            }

            item! {
                #[test]
                fn [<test_2d_zero_like_2_ $t>]() {
                    let a = (array![[42 as $t, 42 as $t], [42 as $t, 42 as $t]]).zero_like();
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
