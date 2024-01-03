// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminDiv;
use ndarray::{Array1, Array2};
use num_complex::Complex;

macro_rules! make_div {
    ($t:ty) => {
        impl ArgminDiv<$t, Array1<$t>> for Array1<$t> {
            #[inline]
            fn div(&self, other: &$t) -> Array1<$t> {
                self / *other
            }
        }

        impl ArgminDiv<Array1<$t>, Array1<$t>> for $t {
            #[inline]
            fn div(&self, other: &Array1<$t>) -> Array1<$t> {
                *self / other
            }
        }

        impl ArgminDiv<Array1<$t>, Array1<$t>> for Array1<$t> {
            #[inline]
            fn div(&self, other: &Array1<$t>) -> Array1<$t> {
                self / other
            }
        }

        impl ArgminDiv<Array2<$t>, Array2<$t>> for Array2<$t> {
            #[inline]
            fn div(&self, other: &Array2<$t>) -> Array2<$t> {
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
make_div!(Complex<f32>);
make_div!(Complex<f64>);

// All code that does not depend on a linked ndarray-linalg backend can still be tested as normal.
// To avoid dublicating tests and to allow convenient testing of functionality that does not need ndarray-linalg the tests are still included here.
// The tests expect the name for the crate containing the tested functions to be argmin_math
#[cfg(test)]
use crate as argmin_math;
include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/ndarray-tests-src/div.rs"
));
