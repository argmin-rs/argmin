// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminConj;
use num_complex::Complex;

macro_rules! make_conj {
    ($t:ty) => {
        impl ArgminConj for $t {
            #[inline]
            fn conj(&self) -> $t {
                *self
            }
        }
    };
}

macro_rules! make_complex_conj {
    ($t:ty) => {
        impl ArgminConj for $t {
            #[inline]
            fn conj(&self) -> $t {
                Complex::conj(self)
            }
        }
    };
}

make_conj!(i8);
make_conj!(i16);
make_conj!(i32);
make_conj!(i64);
make_conj!(u8);
make_conj!(u16);
make_conj!(u32);
make_conj!(u64);
make_conj!(f32);
make_conj!(f64);
make_complex_conj!(Complex<i8>);
make_complex_conj!(Complex<i16>);
make_complex_conj!(Complex<i32>);
make_complex_conj!(Complex<i64>);
make_complex_conj!(Complex<f32>);
make_complex_conj!(Complex<f64>);

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use paste::item;

    macro_rules! make_test_complex {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_complex_conj_ $t>]() {
                    let a = 8 as $t;
                    let b = 34 as $t;
                    let res = <Complex<$t> as ArgminConj>::conj(&Complex::new(a, b));
                    assert_relative_eq!(a as f64, res.re as f64, epsilon = std::f64::EPSILON);
                    assert_relative_eq!(-b as f64, res.im as f64, epsilon = std::f64::EPSILON);
                }
            }
        };
    }

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_conj_ $t>]() {
                    let a = 8 as $t;
                    let res = <$t as ArgminConj>::conj(&a);
                    assert_relative_eq!(a as f64, res as f64, epsilon = std::f64::EPSILON);
                }
            }
        };
    }

    make_test_complex!(i8);
    make_test_complex!(i16);
    make_test_complex!(i32);
    make_test_complex!(i64);
    make_test_complex!(f32);
    make_test_complex!(f64);
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
