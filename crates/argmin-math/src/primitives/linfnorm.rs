// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminLInfNorm;
use num_complex::Complex;

macro_rules! make_linfnorm_unsigned {
    ($t:ty) => {
        impl ArgminLInfNorm<$t> for $t {
            #[inline]
            fn linf_norm(&self) -> $t {
                *self
            }
        }
    };
}

macro_rules! make_linfnorm {
    ($t:ty) => {
        impl ArgminLInfNorm<$t> for $t {
            #[inline]
            fn linf_norm(&self) -> $t {
                self.abs()
            }
        }
    };
}

macro_rules! make_linfnorm_complex {
    ($t:ty) => {
        impl ArgminLInfNorm<$t> for Complex<$t> {
            #[inline]
            fn linf_norm(&self) -> $t {
                (*self).re.hypot((*self).im)
            }
        }
    };
}

make_linfnorm!(i8);
make_linfnorm!(i16);
make_linfnorm!(i32);
make_linfnorm!(i64);
make_linfnorm_unsigned!(u8);
make_linfnorm_unsigned!(u16);
make_linfnorm_unsigned!(u32);
make_linfnorm_unsigned!(u64);
make_linfnorm!(f32);
make_linfnorm!(f64);

make_linfnorm_complex!(f32);
make_linfnorm_complex!(f64);

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_linfnorm_ $t>]() {
                    let a = 8 as $t;
                    let res = <$t as ArgminLInfNorm<$t>>::linf_norm(&a);
                    assert_relative_eq!(a as f64, res as f64, epsilon = f64::EPSILON);
                }
            }
        };
    }

    macro_rules! make_test_signed {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_linfnorm_signed_ $t>]() {
                    let a = -8 as $t;
                    let res = <$t as ArgminLInfNorm<$t>>::linf_norm(&a);
                    assert_relative_eq!(8 as f64, res as f64, epsilon = f64::EPSILON);
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

    make_test_signed!(i8);
    make_test_signed!(i16);
    make_test_signed!(i32);
    make_test_signed!(i64);
    make_test_signed!(f32);
    make_test_signed!(f64);
}
