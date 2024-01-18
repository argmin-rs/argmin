// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminDot;
use num_complex::Complex;

macro_rules! make_dot_vec {
    ($t:ty) => {
        impl ArgminDot<$t, $t> for $t {
            #[inline]
            fn dot(&self, other: &$t) -> $t {
                self * other
            }
        }
    };
}

make_dot_vec!(f32);
make_dot_vec!(f64);
make_dot_vec!(i8);
make_dot_vec!(i16);
make_dot_vec!(i32);
make_dot_vec!(i64);
make_dot_vec!(u8);
make_dot_vec!(u16);
make_dot_vec!(u32);
make_dot_vec!(u64);
make_dot_vec!(isize);
make_dot_vec!(usize);
make_dot_vec!(Complex<f32>);
make_dot_vec!(Complex<f64>);
make_dot_vec!(Complex<i8>);
make_dot_vec!(Complex<i16>);
make_dot_vec!(Complex<i32>);
make_dot_vec!(Complex<i64>);
make_dot_vec!(Complex<u8>);
make_dot_vec!(Complex<u16>);
make_dot_vec!(Complex<u32>);
make_dot_vec!(Complex<u64>);
make_dot_vec!(Complex<isize>);
make_dot_vec!(Complex<usize>);

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_vec_vec_ $t>]() {
                    let a = 21 as $t;
                    let b = 2 as $t;
                    let res = a.dot(&b);
                    assert_relative_eq!(42 as f64, res as f64, epsilon = std::f64::EPSILON);
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
