// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminMul;
use ndarray::{Array1, Array2};
use num_complex::Complex;

macro_rules! make_mul {
    ($t:ty) => {
        impl ArgminMul<$t, Array1<$t>> for Array1<$t> {
            #[inline]
            fn mul(&self, other: &$t) -> Array1<$t> {
                cfg_if::cfg_if! {
                    if #[cfg(feature = "ndarray_0_14")] {
                        self.iter().map(|s| s * other).collect()
                    } else if #[cfg(feature = "ndarray_0_13")]  {
                        self.iter().map(|s| s * other).collect()
                    } else {
                        self * *other
                    }
                }
            }
        }

        impl ArgminMul<Array1<$t>, Array1<$t>> for $t {
            #[inline]
            fn mul(&self, other: &Array1<$t>) -> Array1<$t> {
                cfg_if::cfg_if! {
                    if #[cfg(feature = "ndarray_0_14")] {
                        other.iter().map(|o| o * *self).collect()
                    } else if #[cfg(feature = "ndarray_0_13")]  {
                        other.iter().map(|o| o * *self).collect()
                    } else {
                        *self * other
                    }
                }
            }
        }

        impl ArgminMul<Array1<$t>, Array1<$t>> for Array1<$t> {
            #[inline]
            fn mul(&self, other: &Array1<$t>) -> Array1<$t> {
                cfg_if::cfg_if! {
                    if #[cfg(feature = "ndarray_0_14")] {
                        // Need to assert that the shapes are the same here because the iterators
                        // will silently truncate.
                        assert_eq!(self.shape(), other.shape());
                        self.iter().zip(other.iter()).map(|(s, o)| s * o).collect()
                    } else if #[cfg(feature = "ndarray_0_13")]  {
                        // Need to assert that the shapes are the same here because the iterators
                        // will silently truncate.
                        assert_eq!(self.shape(), other.shape());
                        self.iter().zip(other.iter()).map(|(s, o)| s * o).collect()
                    } else {
                        self * other
                    }
                }
            }
        }

        impl ArgminMul<Array2<$t>, Array2<$t>> for Array2<$t> {
            #[inline]
            fn mul(&self, other: &Array2<$t>) -> Array2<$t> {
                self * other
            }
        }

        impl ArgminMul<$t, Array2<$t>> for Array2<$t> {
            #[inline]
            fn mul(&self, other: &$t) -> Array2<$t> {
                self * *other
            }
        }

        impl ArgminMul<Array2<$t>, Array2<$t>> for $t {
            #[inline]
            fn mul(&self, other: &Array2<$t>) -> Array2<$t> {
                other * *self
            }
        }
    };
}

macro_rules! make_complex_mul {
    ($t:ty) => {
        impl ArgminMul<Complex<$t>, Array1<Complex<$t>>> for Array1<Complex<$t>> {
            #[inline]
            fn mul(&self, other: &Complex<$t>) -> Array1<Complex<$t>> {
                self * *other
            }
        }

        impl ArgminMul<$t, Array1<Complex<$t>>> for Array1<Complex<$t>> {
            #[inline]
            fn mul(&self, other: &$t) -> Array1<Complex<$t>> {
                self * *other
            }
        }

        impl ArgminMul<Array1<Complex<$t>>, Array1<Complex<$t>>> for Complex<$t> {
            #[inline]
            fn mul(&self, other: &Array1<Complex<$t>>) -> Array1<Complex<$t>> {
                *self * other
            }
        }

        impl ArgminMul<Array1<Complex<$t>>, Array1<Complex<$t>>> for $t {
            #[inline]
            fn mul(&self, other: &Array1<Complex<$t>>) -> Array1<Complex<$t>> {
                Complex::new(*self, 0 as $t) * other
            }
        }

        impl ArgminMul<Array1<Complex<$t>>, Array1<Complex<$t>>> for Array1<Complex<$t>> {
            #[inline]
            fn mul(&self, other: &Array1<Complex<$t>>) -> Array1<Complex<$t>> {
                self * other
            }
        }

        impl ArgminMul<Array2<Complex<$t>>, Array2<Complex<$t>>> for Array2<Complex<$t>> {
            #[inline]
            fn mul(&self, other: &Array2<Complex<$t>>) -> Array2<Complex<$t>> {
                self * other
            }
        }

        impl ArgminMul<Complex<$t>, Array2<Complex<$t>>> for Array2<Complex<$t>> {
            #[inline]
            fn mul(&self, other: &Complex<$t>) -> Array2<Complex<$t>> {
                self * *other
            }
        }

        impl ArgminMul<$t, Array2<Complex<$t>>> for Array2<Complex<$t>> {
            #[inline]
            fn mul(&self, other: &$t) -> Array2<Complex<$t>> {
                self * *other
            }
        }

        impl ArgminMul<Array2<Complex<$t>>, Array2<Complex<$t>>> for Complex<$t> {
            #[inline]
            fn mul(&self, other: &Array2<Complex<$t>>) -> Array2<Complex<$t>> {
                *self * other
            }
        }

        impl ArgminMul<Array2<Complex<$t>>, Array2<Complex<$t>>> for $t {
            #[inline]
            fn mul(&self, other: &Array2<Complex<$t>>) -> Array2<Complex<$t>> {
                Complex::new(*self, 0 as $t) * other
            }
        }
    };
}

macro_rules! make_complex_integer_mul {
    ($t:ty) => {
        impl ArgminMul<Complex<$t>, Array1<Complex<$t>>> for Array1<Complex<$t>> {
            #[inline]
            fn mul(&self, other: &Complex<$t>) -> Array1<Complex<$t>> {
                self.iter().map(|s| s * *other).collect()
            }
        }

        impl ArgminMul<$t, Array1<Complex<$t>>> for Array1<Complex<$t>> {
            #[inline]
            fn mul(&self, other: &$t) -> Array1<Complex<$t>> {
                self.iter().map(|s| s * *other).collect()
            }
        }

        impl ArgminMul<Array1<Complex<$t>>, Array1<Complex<$t>>> for Complex<$t> {
            #[inline]
            fn mul(&self, other: &Array1<Complex<$t>>) -> Array1<Complex<$t>> {
                other.iter().map(|o| o * *self).collect()
            }
        }

        impl ArgminMul<Array1<Complex<$t>>, Array1<Complex<$t>>> for $t {
            #[inline]
            fn mul(&self, other: &Array1<Complex<$t>>) -> Array1<Complex<$t>> {
                let s = Complex::new(*self, 0 as $t);
                other.iter().map(|o| o * s).collect()
            }
        }

        impl ArgminMul<Array1<Complex<$t>>, Array1<Complex<$t>>> for Array1<Complex<$t>> {
            #[inline]
            fn mul(&self, other: &Array1<Complex<$t>>) -> Array1<Complex<$t>> {
                self * other
            }
        }

        impl ArgminMul<Array2<Complex<$t>>, Array2<Complex<$t>>> for Array2<Complex<$t>> {
            #[inline]
            fn mul(&self, other: &Array2<Complex<$t>>) -> Array2<Complex<$t>> {
                self * other
            }
        }

        impl ArgminMul<Complex<$t>, Array2<Complex<$t>>> for Array2<Complex<$t>> {
            #[inline]
            fn mul(&self, other: &Complex<$t>) -> Array2<Complex<$t>> {
                Array2::from_shape_fn((self.nrows(), self.ncols()), |(i, j)| self[(i, j)] * *other)
            }
        }

        impl ArgminMul<$t, Array2<Complex<$t>>> for Array2<Complex<$t>> {
            #[inline]
            fn mul(&self, other: &$t) -> Array2<Complex<$t>> {
                self * *other
            }
        }

        impl ArgminMul<Array2<Complex<$t>>, Array2<Complex<$t>>> for Complex<$t> {
            #[inline]
            fn mul(&self, other: &Array2<Complex<$t>>) -> Array2<Complex<$t>> {
                Array2::from_shape_fn((other.nrows(), other.ncols()), |(i, j)| {
                    other[(i, j)] * *self
                })
            }
        }

        impl ArgminMul<Array2<Complex<$t>>, Array2<Complex<$t>>> for $t {
            #[inline]
            fn mul(&self, other: &Array2<Complex<$t>>) -> Array2<Complex<$t>> {
                Array2::from_shape_fn((other.nrows(), other.ncols()), |(i, j)| {
                    other[(i, j)] * *self
                })
            }
        }
    };
}

make_mul!(i8);
make_mul!(u8);
make_mul!(i16);
make_mul!(u16);
make_mul!(i32);
make_mul!(u32);
make_mul!(i64);
make_mul!(u64);
make_mul!(isize);
make_mul!(usize);
make_mul!(f32);
make_mul!(f64);
make_complex_mul!(f32);
make_complex_mul!(f64);
make_complex_integer_mul!(i8);
make_complex_integer_mul!(u8);
make_complex_integer_mul!(i16);
make_complex_integer_mul!(u16);
make_complex_integer_mul!(i32);
make_complex_integer_mul!(u32);
make_complex_integer_mul!(i64);
make_complex_integer_mul!(u64);
make_complex_integer_mul!(isize);
make_complex_integer_mul!(usize);

// The tests expect the name for the crate containing the tested functions to be argmin_math
#[cfg(test)]
use crate as argmin_math;
include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../argmin-math-ndarray-linalg-tests/src/mul.rs"
));
