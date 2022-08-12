// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminSignum;
use num_complex::Complex;

macro_rules! make_signum {
    ($t:ty) => {
        impl ArgminSignum for Vec<$t> {
            fn signum(mut self) -> Self {
                for x in &mut self {
                    *x = x.signum();
                }
                self
            }
        }
    };
}

macro_rules! make_signum_complex {
    ($t:ty) => {
        impl ArgminSignum for Vec<$t> {
            fn signum(mut self) -> Self {
                for x in &mut self {
                    x.re = x.re.signum();
                    x.im = x.im.signum();
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
