// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminDiv;
use num_complex::Complex;

macro_rules! make_div {
    ($t:ty) => {
        impl ArgminDiv<$t, $t> for $t {
            #[inline]
            fn div(&self, other: &$t) -> $t {
                self / other
            }
        }
    };
}

make_div!(i8);
make_div!(u8);
make_div!(i16);
make_div!(u16);
make_div!(i32);
make_div!(u32);
make_div!(i64);
make_div!(u64);
make_div!(f32);
make_div!(f64);
make_div!(Complex<i8>);
make_div!(Complex<u8>);
make_div!(Complex<i16>);
make_div!(Complex<u16>);
make_div!(Complex<i32>);
make_div!(Complex<u32>);
make_div!(Complex<i64>);
make_div!(Complex<u64>);
make_div!(Complex<f32>);
make_div!(Complex<f64>);

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_div_ $t>]() {
                    let a = 84 as $t;
                    let b = 2 as $t;
                    let res = <$t as ArgminDiv<$t, $t>>::div(&a, &b);
                    assert_relative_eq!(42 as f64, res as f64, epsilon = std::f64::EPSILON);
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
