// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

// Note: This is not really the preferred way I think. Maybe this should also be implemented for
// ArrayViews, which would probably make it more efficient.

use crate::ArgminTranspose;
use ndarray::{Array1, Array2};
use num_complex::Complex;

macro_rules! make_add {
    ($t:ty) => {
        impl ArgminTranspose<Array1<$t>> for Array1<$t> {
            #[inline]
            fn t(self) -> Array1<$t> {
                self.reversed_axes()
            }
        }

        impl ArgminTranspose<Array2<$t>> for Array2<$t> {
            #[inline]
            fn t(self) -> Array2<$t> {
                self.reversed_axes()
            }
        }
    };
}

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
make_add!(Complex<f32>);
make_add!(Complex<f64>);

// The tests expect the name for the crate containing the tested functions to be argmin_math
#[cfg(test)]
use crate as argmin_math;
include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../argmin-math-ndarray-linalg-tests/src/transpose.rs"
));
