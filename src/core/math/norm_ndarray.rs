// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::math::ArgminNorm;
use ndarray::Array1;
use num_complex::Complex;
use num_integer::Roots;

macro_rules! make_norm_float {
    ($t:ty) => {
        impl ArgminNorm<$t> for Array1<$t> {
            #[inline]
            fn norm(&self) -> $t {
                self.iter().map(|a| a.powi(2)).sum::<$t>().sqrt()
            }
        }
    };
}

macro_rules! make_norm_complex_float {
    ($t:ty) => {
        impl ArgminNorm<Complex<$t>> for Array1<Complex<$t>> {
            #[inline]
            fn norm(&self) -> Complex<$t> {
                self.iter().map(|a| a.powf(2.0)).sum::<Complex<$t>>().sqrt()
            }
        }

        impl ArgminNorm<$t> for Array1<Complex<$t>> {
            #[inline]
            fn norm(&self) -> $t {
                self.iter()
                    .map(|a| a.powf(2.0))
                    .sum::<Complex<$t>>()
                    .sqrt()
                    .norm()
            }
        }
    };
}

macro_rules! make_norm_integer {
    ($t:ty) => {
        impl ArgminNorm<$t> for Array1<$t> {
            #[inline]
            fn norm(&self) -> $t {
                self.iter().map(|a| a.pow(2)).sum::<$t>().sqrt()
            }
        }
    };
}

make_norm_integer!(isize);
make_norm_integer!(usize);
make_norm_integer!(i8);
make_norm_integer!(i16);
make_norm_integer!(i32);
make_norm_integer!(i64);
make_norm_integer!(u8);
make_norm_integer!(u16);
make_norm_integer!(u32);
make_norm_integer!(u64);
make_norm_float!(f32);
make_norm_float!(f64);
make_norm_complex_float!(f32);
make_norm_complex_float!(f64);

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1};
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_norm_ $t>]() {
                    let a = array![4 as $t, 3 as $t];
                    let res = <Array1<$t> as ArgminNorm<$t>>::norm(&a);
                    let target = 5 as $t;
                    assert!(((target - res) as f64).abs() < std::f64::EPSILON);
                }
            }
        };
    }

    macro_rules! make_test_signed {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_norm_signed_ $t>]() {
                    let a = array![-4 as $t, -3 as $t];
                    let res = <Array1<$t> as ArgminNorm<$t>>::norm(&a);
                    let target = 5 as $t;
                    assert!(((target - res) as f64).abs() < std::f64::EPSILON);
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
