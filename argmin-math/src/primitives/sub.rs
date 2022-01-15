// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminSub;
use num_complex::Complex;

macro_rules! make_sub {
    ($t:ty) => {
        impl ArgminSub<$t, $t> for $t {
            #[inline]
            fn sub(&self, other: &$t) -> $t {
                self - other
            }
        }
    };
}

make_sub!(isize);
make_sub!(usize);
make_sub!(i8);
make_sub!(i16);
make_sub!(i32);
make_sub!(i64);
make_sub!(u8);
make_sub!(u16);
make_sub!(u32);
make_sub!(u64);
make_sub!(f32);
make_sub!(f64);
make_sub!(Complex<isize>);
make_sub!(Complex<usize>);
make_sub!(Complex<i8>);
make_sub!(Complex<i16>);
make_sub!(Complex<i32>);
make_sub!(Complex<i64>);
make_sub!(Complex<u8>);
make_sub!(Complex<u16>);
make_sub!(Complex<u32>);
make_sub!(Complex<u64>);
make_sub!(Complex<f32>);
make_sub!(Complex<f64>);

#[cfg(test)]
mod tests {
    use super::*;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_sub_ $t>]() {
                    let a = 50 as $t;
                    let b = 8 as $t;
                    let res = <$t as ArgminSub<$t, $t>>::sub(&a, &b);
                    assert!(((42 as $t - res) as f64).abs() < std::f64::EPSILON);
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
