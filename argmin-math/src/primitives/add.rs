// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminAdd;
use num_complex::Complex;

macro_rules! make_add {
    ($t:ty) => {
        impl ArgminAdd<$t, $t> for $t {
            #[inline]
            fn add(&self, other: &$t) -> $t {
                self + other
            }
        }
    };
}

make_add!(isize);
make_add!(usize);
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
make_add!(Complex<isize>);
make_add!(Complex<usize>);
make_add!(Complex<i8>);
make_add!(Complex<i16>);
make_add!(Complex<i32>);
make_add!(Complex<i64>);
make_add!(Complex<u8>);
make_add!(Complex<u16>);
make_add!(Complex<u32>);
make_add!(Complex<u64>);
make_add!(Complex<f32>);
make_add!(Complex<f64>);

#[cfg(test)]
mod tests {
    use super::*;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_add_ $t>]() {
                    let a = 8 as $t;
                    let b = 34 as $t;
                    let res = <$t as ArgminAdd<$t, $t>>::add(&a, &b);
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
