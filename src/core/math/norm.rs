// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::math::ArgminNorm;
use num_complex::Complex;

macro_rules! make_norm_unsigned {
    ($t:ty) => {
        impl ArgminNorm<$t> for $t {
            #[inline]
            fn norm(&self) -> $t {
                *self
            }
        }
    };
}

macro_rules! make_norm {
    ($t:ty) => {
        impl ArgminNorm<$t> for $t {
            #[inline]
            fn norm(&self) -> $t {
                (*self).abs()
            }
        }
    };
}

macro_rules! make_norm_complex {
    ($t:ty) => {
        impl ArgminNorm<$t> for Complex<$t> {
            #[inline]
            fn norm(&self) -> $t {
                (*self).re.hypot((*self).im)
            }
        }
    };
}

make_norm!(isize);
make_norm_unsigned!(usize);
make_norm!(i8);
make_norm!(i16);
make_norm!(i32);
make_norm!(i64);
make_norm_unsigned!(u8);
make_norm_unsigned!(u16);
make_norm_unsigned!(u32);
make_norm_unsigned!(u64);
make_norm!(f32);
make_norm!(f64);

make_norm_complex!(f32);
make_norm_complex!(f64);

#[cfg(test)]
mod tests {
    use super::*;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_norm_ $t>]() {
                    let a = 8 as $t;
                    let res = <$t as ArgminNorm<$t>>::norm(&a);
                    assert!(((a - res) as f64).abs() < std::f64::EPSILON);
                }
            }
        };
    }

    macro_rules! make_test_signed {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_norm_signed_ $t>]() {
                    let a = -8 as $t;
                    let res = <$t as ArgminNorm<$t>>::norm(&a);
                    assert!(((8 as $t - res) as f64).abs() < std::f64::EPSILON);
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

    make_test_signed!(isize);
    make_test_signed!(i8);
    make_test_signed!(i16);
    make_test_signed!(i32);
    make_test_signed!(i64);
    make_test_signed!(f32);
    make_test_signed!(f64);
}
