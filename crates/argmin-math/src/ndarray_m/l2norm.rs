// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminL2Norm;
use ndarray::Array1;
use num_complex::Complex;
use num_integer::Roots;

macro_rules! make_norm_float {
    ($t:ty) => {
        impl ArgminL2Norm<$t> for Array1<$t> {
            #[inline]
            fn l2_norm(&self) -> $t {
                self.iter().map(|a| a.powi(2)).sum::<$t>().sqrt()
            }
        }
    };
}

macro_rules! make_norm_integer {
    ($t:ty) => {
        impl ArgminL2Norm<$t> for Array1<$t> {
            #[inline]
            fn l2_norm(&self) -> $t {
                self.iter().map(|a| a.pow(2)).sum::<$t>().sqrt()
            }
        }
    };
}

macro_rules! make_norm_complex {
    ($i: ty, $t:ty) => {
        impl ArgminL2Norm<$t> for Array1<$i> {
            #[inline]
            fn l2_norm(&self) -> $t {
                self.iter().map(|a| a.norm_sqr()).sum::<$t>().sqrt()
            }
        }
    };
}

macro_rules! make_norm_unsigned {
    ($t:ty) => {
        impl ArgminL2Norm<$t> for Array1<$t> {
            #[inline]
            fn l2_norm(&self) -> $t {
                self.iter().map(|a| a.pow(2)).sum::<$t>().sqrt()
            }
        }
    };
}

make_norm_unsigned!(u8);
make_norm_unsigned!(u16);
make_norm_unsigned!(u32);
make_norm_unsigned!(u64);
make_norm_integer!(i8);
make_norm_integer!(i16);
make_norm_integer!(i32);
make_norm_integer!(i64);
make_norm_float!(f32);
make_norm_float!(f64);
make_norm_complex!(Complex<i8>, i8);
make_norm_complex!(Complex<i16>, i16);
make_norm_complex!(Complex<i32>, i32);
make_norm_complex!(Complex<i64>, i64);
make_norm_complex!(Complex<u8>, u8);
make_norm_complex!(Complex<u16>, u16);
make_norm_complex!(Complex<u32>, u32);
make_norm_complex!(Complex<u64>, u64);
make_norm_complex!(Complex<f32>, f32);
make_norm_complex!(Complex<f64>, f64);

// All code that does not depend on a linked ndarray-linalg backend can still be tested as normal.
// To avoid dublicating tests and to allow convenient testing of functionality that does not need ndarray-linalg the tests are still included here.
// The tests expect the name for the crate containing the tested functions to be argmin_math
#[cfg(test)]
use crate as argmin_math;
include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/ndarray-tests-src/l2norm.rs"
));
