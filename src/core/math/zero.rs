// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::math::{ArgminZero, ArgminZeroLike};
use num_complex::Complex;

macro_rules! make_zero {
    ($t:ty) => {
        impl ArgminZero for $t {
            #[allow(clippy::cast_lossless)]
            #[inline]
            fn zero() -> $t {
                0 as $t
            }
        }
        impl ArgminZeroLike for $t {
            #[allow(clippy::cast_lossless)]
            #[inline]
            fn zero_like(&self) -> $t {
                0 as $t
            }
        }
    };
}

macro_rules! make_complex_zero {
    ($t:ty) => {
        impl ArgminZero for Complex<$t> {
            #[allow(clippy::cast_lossless)]
            #[inline]
            fn zero() -> Complex<$t> {
                Complex::new(0 as $t, 0 as $t)
            }
        }
        impl ArgminZeroLike for Complex<$t> {
            #[allow(clippy::cast_lossless)]
            #[inline]
            fn zero_like(&self) -> Complex<$t> {
                Complex::new(0 as $t, 0 as $t)
            }
        }
    };
}

make_zero!(f32);
make_zero!(f64);
make_zero!(i8);
make_zero!(i16);
make_zero!(i32);
make_zero!(i64);
make_zero!(u8);
make_zero!(u16);
make_zero!(u32);
make_zero!(u64);
make_zero!(isize);
make_zero!(usize);
make_complex_zero!(f32);
make_complex_zero!(f64);
make_complex_zero!(i8);
make_complex_zero!(i16);
make_complex_zero!(i32);
make_complex_zero!(i64);
make_complex_zero!(u8);
make_complex_zero!(u16);
make_complex_zero!(u32);
make_complex_zero!(u64);
make_complex_zero!(isize);
make_complex_zero!(usize);

#[cfg(test)]
mod tests {
    use super::*;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_zero_ $t>]() {
                    let a = <$t as ArgminZero>::zero();
                    assert!(((0 as $t - a) as f64).abs() < std::f64::EPSILON);
                }
            }

            item! {
                #[test]
                fn [<test_zero_like_ $t>]() {
                    let a = (42 as $t).zero_like();
                    assert!(((0 as $t - a) as f64).abs() < std::f64::EPSILON);
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
