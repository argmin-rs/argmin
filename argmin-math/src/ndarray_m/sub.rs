// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminSub;
use ndarray::{Array1, Array2};
use num_complex::Complex;

macro_rules! make_sub {
    ($t:ty) => {
        impl ArgminSub<$t, Array1<$t>> for Array1<$t> {
            #[inline]
            fn sub(&self, other: &$t) -> Array1<$t> {
                self - *other
            }
        }

        impl ArgminSub<Array1<$t>, Array1<$t>> for $t {
            #[inline]
            fn sub(&self, other: &Array1<$t>) -> Array1<$t> {
                *self - other
            }
        }

        impl ArgminSub<Array1<$t>, Array1<$t>> for Array1<$t> {
            #[inline]
            fn sub(&self, other: &Array1<$t>) -> Array1<$t> {
                self - other
            }
        }

        impl ArgminSub<Array2<$t>, Array2<$t>> for Array2<$t> {
            #[inline]
            fn sub(&self, other: &Array2<$t>) -> Array2<$t> {
                self - other
            }
        }

        impl ArgminSub<$t, Array2<$t>> for Array2<$t> {
            #[inline]
            fn sub(&self, other: &$t) -> Array2<$t> {
                self - *other
            }
        }
    };
}

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
make_sub!(Complex<f32>);
make_sub!(Complex<f64>);

// The tests expect the name for the crate containing the tested functions to be argmin_math
#[cfg(test)]
use crate as argmin_math;
include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../argmin-math-ndarray-linalg-tests/src/sub.rs"
));
