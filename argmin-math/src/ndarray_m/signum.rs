// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminSignum;
use ndarray::{Array1, Array2};
use num_complex::Complex;

macro_rules! make_signum {
    ($t:ty) => {
        impl ArgminSignum for Array1<$t> {
            #[inline]
            fn signum(mut self) -> Array1<$t> {
                for a in &mut self {
                    *a = a.signum();
                }
                self
            }
        }

        impl ArgminSignum for Array2<$t> {
            #[inline]
            fn signum(mut self) -> Array2<$t> {
                let m = self.shape()[0];
                let n = self.shape()[1];
                for i in 0..m {
                    for j in 0..n {
                        self[(i, j)] = self[(i, j)].signum();
                    }
                }
                self
            }
        }
    };
}

macro_rules! make_signum_complex {
    ($t:ty) => {
        impl ArgminSignum for Array1<$t> {
            #[inline]
            fn signum(mut self) -> Array1<$t> {
                for a in &mut self {
                    a.re = a.re.signum();
                    a.im = a.im.signum();
                }
                self
            }
        }

        impl ArgminSignum for Array2<$t> {
            #[inline]
            fn signum(mut self) -> Array2<$t> {
                let m = self.shape()[0];
                let n = self.shape()[1];
                for i in 0..m {
                    for j in 0..n {
                        self[(i, j)].re = self[(i, j)].re.signum();
                        self[(i, j)].im = self[(i, j)].im.signum();
                    }
                }
                self
            }
        }
    };
}

make_signum!(isize);
make_signum!(i8);
make_signum!(i16);
make_signum!(i32);
make_signum!(i64);
make_signum!(f32);
make_signum!(f64);
make_signum_complex!(Complex<isize>);
make_signum_complex!(Complex<i8>);
make_signum_complex!(Complex<i16>);
make_signum_complex!(Complex<i32>);
make_signum_complex!(Complex<i64>);
make_signum_complex!(Complex<f32>);
make_signum_complex!(Complex<f64>);

// The tests expect the name for the crate containing the tested functions to be argmin_math
#[cfg(test)]
use crate as argmin_math;
include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/ndarray-tests-src/signum.rs"
));
