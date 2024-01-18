// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminEye;
use ndarray::Array2;

macro_rules! make_eye {
    ($t:ty) => {
        impl ArgminEye for Array2<$t> {
            #[inline]
            fn eye_like(&self) -> Array2<$t> {
                // TODO: Should return an error!
                assert!(self.is_square());
                ndarray::Array2::eye(self.dim().0)
            }

            #[inline]
            fn eye(n: usize) -> Array2<$t> {
                ndarray::Array2::eye(n)
            }
        }
    };
}

make_eye!(isize);
make_eye!(usize);
make_eye!(i8);
make_eye!(i16);
make_eye!(i32);
make_eye!(i64);
make_eye!(u8);
make_eye!(u16);
make_eye!(u32);
make_eye!(u64);
make_eye!(f32);
make_eye!(f64);

// All code that does not depend on a linked ndarray-linalg backend can still be tested as normal.
// To avoid dublicating tests and to allow convenient testing of functionality that does not need ndarray-linalg the tests are still included here.
// The tests expect the name for the crate containing the tested functions to be argmin_math
#[cfg(test)]
use crate as argmin_math;
include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/ndarray-tests-src/eye.rs"
));
