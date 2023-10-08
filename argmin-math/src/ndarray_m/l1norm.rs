// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminL1Norm;
use ndarray::Array1;
use num_complex::Complex;

macro_rules! make_l1norm {
    ($t:ty) => {
        impl ArgminL1Norm<$t> for Array1<$t> {
            #[inline]
            fn l1_norm(&self) -> $t {
                self.iter().map(|a| a.abs()).sum()
            }
        }
    };
}

macro_rules! make_l1norm_complex {
    ($i: ty, $t:ty) => {
        impl ArgminL1Norm<$t> for Array1<$i> {
            #[inline]
            fn l1_norm(&self) -> $t {
                self.iter().map(|a| a.l1_norm()).sum::<$t>().into()
            }
        }
    };
}

macro_rules! make_l1norm_unsigned {
    ($t:ty) => {
        impl ArgminL1Norm<$t> for Array1<$t> {
            #[inline]
            fn l1_norm(&self) -> $t {
                self.iter().sum()
            }
        }
    };
}

make_l1norm_unsigned!(u8);
make_l1norm_unsigned!(u16);
make_l1norm_unsigned!(u32);
make_l1norm_unsigned!(u64);
make_l1norm_unsigned!(usize);
make_l1norm!(i8);
make_l1norm!(i16);
make_l1norm!(i32);
make_l1norm!(i64);
make_l1norm!(isize);
make_l1norm!(f32);
make_l1norm!(f64);
make_l1norm_complex!(Complex<i8>, i8);
make_l1norm_complex!(Complex<i16>, i16);
make_l1norm_complex!(Complex<i32>, i32);
make_l1norm_complex!(Complex<i64>, i64);
make_l1norm_complex!(Complex<isize>, isize);
make_l1norm_complex!(Complex<u8>, u8);
make_l1norm_complex!(Complex<u16>, u16);
make_l1norm_complex!(Complex<u32>, u32);
make_l1norm_complex!(Complex<u64>, u64);
make_l1norm_complex!(Complex<usize>, usize);
make_l1norm_complex!(Complex<f32>, f32);
make_l1norm_complex!(Complex<f64>, f64);

// The tests expect the name for the crate containing the tested functions to be argmin_math
#[cfg(test)]
use crate as argmin_math;
include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/ndarray-tests-src/l1norm.rs"
));
