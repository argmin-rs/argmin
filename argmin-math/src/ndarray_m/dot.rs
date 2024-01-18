// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminDot;
use ndarray::{Array1, Array2};
use num_complex::Complex;

macro_rules! make_dot_ndarray {
    ($t:ty) => {
        impl ArgminDot<Array1<$t>, $t> for Array1<$t> {
            #[inline]
            fn dot(&self, other: &Array1<$t>) -> $t {
                ndarray::Array1::dot(self, other)
            }
        }

        impl ArgminDot<$t, Array1<$t>> for Array1<$t> {
            #[inline]
            fn dot(&self, other: &$t) -> Array1<$t> {
                self.iter().cloned().map(|s| s * *other).collect()
            }
        }

        impl ArgminDot<Array1<$t>, Array1<$t>> for $t {
            #[inline]
            fn dot(&self, other: &Array1<$t>) -> Array1<$t> {
                other.iter().cloned().map(|o| o * *self).collect()
            }
        }

        impl ArgminDot<Array1<$t>, Array2<$t>> for Array1<$t> {
            #[inline]
            fn dot(&self, other: &Array1<$t>) -> Array2<$t> {
                Array2::from_shape_fn((self.len(), other.len()), |(i, j)| self[i] * other[j])
            }
        }

        impl ArgminDot<Array1<$t>, Array1<$t>> for Array2<$t> {
            #[inline]
            fn dot(&self, other: &Array1<$t>) -> Array1<$t> {
                ndarray::Array2::dot(self, other)
            }
        }

        impl ArgminDot<Array2<$t>, Array2<$t>> for Array2<$t> {
            #[inline]
            fn dot(&self, other: &Array2<$t>) -> Array2<$t> {
                ndarray::Array2::dot(self, other)
            }
        }

        impl ArgminDot<$t, Array2<$t>> for Array2<$t> {
            #[inline]
            fn dot(&self, other: &$t) -> Array2<$t> {
                Array2::from_shape_fn((self.nrows(), self.ncols()), |(i, j)| *other * self[(i, j)])
            }
        }

        impl ArgminDot<Array2<$t>, Array2<$t>> for $t {
            #[inline]
            fn dot(&self, other: &Array2<$t>) -> Array2<$t> {
                Array2::from_shape_fn((other.nrows(), other.ncols()), |(i, j)| {
                    *self * other[(i, j)]
                })
            }
        }

        impl ArgminDot<Array1<Complex<$t>>, Complex<$t>> for Array1<Complex<$t>> {
            #[inline]
            fn dot(&self, other: &Array1<Complex<$t>>) -> Complex<$t> {
                ndarray::Array1::dot(self, other)
            }
        }

        impl ArgminDot<Complex<$t>, Array1<Complex<$t>>> for Array1<Complex<$t>> {
            #[inline]
            fn dot(&self, other: &Complex<$t>) -> Array1<Complex<$t>> {
                self.iter().cloned().map(|s| s * *other).collect()
            }
        }

        impl ArgminDot<Array1<Complex<$t>>, Array1<Complex<$t>>> for Complex<$t> {
            #[inline]
            fn dot(&self, other: &Array1<Complex<$t>>) -> Array1<Complex<$t>> {
                other.iter().cloned().map(|o| o * *self).collect()
            }
        }

        impl ArgminDot<Array1<Complex<$t>>, Array2<Complex<$t>>> for Array1<Complex<$t>> {
            #[inline]
            fn dot(&self, other: &Array1<Complex<$t>>) -> Array2<Complex<$t>> {
                Array2::from_shape_fn((self.len(), other.len()), |(i, j)| self[i] * other[j])
            }
        }

        impl ArgminDot<Array1<Complex<$t>>, Array1<Complex<$t>>> for Array2<Complex<$t>> {
            #[inline]
            fn dot(&self, other: &Array1<Complex<$t>>) -> Array1<Complex<$t>> {
                ndarray::Array2::dot(self, other)
            }
        }

        impl ArgminDot<Array2<Complex<$t>>, Array2<Complex<$t>>> for Array2<Complex<$t>> {
            #[inline]
            fn dot(&self, other: &Array2<Complex<$t>>) -> Array2<Complex<$t>> {
                ndarray::Array2::dot(self, other)
            }
        }

        impl ArgminDot<Complex<$t>, Array2<Complex<$t>>> for Array2<Complex<$t>> {
            #[inline]
            fn dot(&self, other: &Complex<$t>) -> Array2<Complex<$t>> {
                Array2::from_shape_fn((self.nrows(), self.ncols()), |(i, j)| *other * self[(i, j)])
            }
        }

        impl ArgminDot<Array2<Complex<$t>>, Array2<Complex<$t>>> for Complex<$t> {
            #[inline]
            fn dot(&self, other: &Array2<Complex<$t>>) -> Array2<Complex<$t>> {
                Array2::from_shape_fn((other.nrows(), other.ncols()), |(i, j)| {
                    *self * other[(i, j)]
                })
            }
        }
    };
}

make_dot_ndarray!(i8);
make_dot_ndarray!(i16);
make_dot_ndarray!(i32);
make_dot_ndarray!(i64);
make_dot_ndarray!(isize);
make_dot_ndarray!(u8);
make_dot_ndarray!(u16);
make_dot_ndarray!(u32);
make_dot_ndarray!(u64);
make_dot_ndarray!(usize);
make_dot_ndarray!(f32);
make_dot_ndarray!(f64);

// All code that does not depend on a linked ndarray-linalg backend can still be tested as normal.
// To avoid dublicating tests and to allow convenient testing of functionality that does not need ndarray-linalg the tests are still included here.
// The tests expect the name for the crate containing the tested functions to be argmin_math
#[cfg(test)]
use crate as argmin_math;
include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/ndarray-tests-src/dot.rs"
));
