// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminLInfNorm;
use ndarray::Array1;
use num_complex::Complex;

macro_rules! make_linfnorm_float {
    ($t:ty) => {
        impl ArgminLInfNorm<$t> for Array1<$t> {
            #[inline]
            fn linf_norm(&self) -> $t {
                self.iter().fold(0 as $t, |m, a| m.max(a.abs()))
            }
        }
    };
}

macro_rules! make_linfnorm_signed_integer {
    ($t:ty) => {
        impl ArgminLInfNorm<$t> for Array1<$t> {
            #[inline]
            fn linf_norm(&self) -> $t {
                self.iter().fold(0 as $t, |m, a| m.max(a.abs()))
            }
        }
    };
}

macro_rules! make_linfnorm_unsigned {
    ($t:ty) => {
        impl ArgminLInfNorm<$t> for Array1<$t> {
            #[inline]
            fn linf_norm(&self) -> $t {
                self.iter().copied().fold(0 as $t, <$t>::max)
            }
        }
    };
}

macro_rules! make_linfnorm_complex {
    ($i:ty, $t:ty) => {
        impl ArgminLInfNorm<$t> for Array1<$i> {
            #[inline]
            fn linf_norm(&self) -> $t {
                self.iter()
                    .map(|a| a.norm_sqr())
                    .fold(0 as $t, <$t>::max)
                    .sqrt()
            }
        }
    };
}

make_linfnorm_unsigned!(u8);
make_linfnorm_unsigned!(u16);
make_linfnorm_unsigned!(u32);
make_linfnorm_unsigned!(u64);
make_linfnorm_signed_integer!(i8);
make_linfnorm_signed_integer!(i16);
make_linfnorm_signed_integer!(i32);
make_linfnorm_signed_integer!(i64);
make_linfnorm_float!(f32);
make_linfnorm_float!(f64);
make_linfnorm_complex!(Complex<f32>, f32);
make_linfnorm_complex!(Complex<f64>, f64);

// All code that does not depend on a linked ndarray-linalg backend can still be tested as normal.
// To avoid duplicating tests and to allow convenient testing of functionality that does not need ndarray-linalg the tests are still included here.
// The tests expect the name for the crate containing the tested functions to be argmin_math
#[cfg(test)]
use crate as argmin_math;
include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/ndarray-tests-src/linfnorm.rs"
));
