// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminMinMax;
use num_complex::Complex;

macro_rules! make_minmax {
    ($t:ty) => {
        impl ArgminMinMax for $t {
            #[inline]
            fn min(x: &Self, y: &Self) -> $t {
                std::cmp::min(x, y)
            }

            fn max(x: &Self, y: &Self) -> $t {
                std::cmp::max(x, y)
            }
        }
    };
}

make_minmax!(f32);
make_minmax!(f64);
make_minmax!(i8);
make_minmax!(i16);
make_minmax!(i32);
make_minmax!(i64);
make_minmax!(u8);
make_minmax!(u16);
make_minmax!(u32);
make_minmax!(u64);
make_minmax!(isize);
make_minmax!(usize);
make_minmax!(Complex<f32>);
make_minmax!(Complex<f64>);
make_minmax!(Complex<i8>);
make_minmax!(Complex<i16>);
make_minmax!(Complex<i32>);
make_minmax!(Complex<i64>);
make_minmax!(Complex<u8>);
make_minmax!(Complex<u16>);
make_minmax!(Complex<u32>);
make_minmax!(Complex<u64>);
make_minmax!(Complex<isize>);
make_minmax!(Complex<usize>);

// TODO: tests!!!
