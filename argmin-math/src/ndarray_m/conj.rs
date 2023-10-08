// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminConj;
use ndarray::{Array1, Array2};
use num_complex::Complex;

macro_rules! make_conj {
    ($t:ty) => {
        impl ArgminConj for Array1<$t> {
            #[inline]
            fn conj(&self) -> Array1<$t> {
                self.iter().map(|a| <$t as ArgminConj>::conj(a)).collect()
            }
        }

        impl ArgminConj for Array2<$t> {
            #[inline]
            fn conj(&self) -> Array2<$t> {
                let n = self.shape();
                let mut out = self.clone();
                for i in 0..n[0] {
                    for j in 0..n[1] {
                        out[(i, j)] = out[(i, j)].conj();
                    }
                }
                out
            }
        }
    };
}

make_conj!(isize);
make_conj!(i8);
make_conj!(i16);
make_conj!(i32);
make_conj!(i64);
make_conj!(f32);
make_conj!(f64);
make_conj!(Complex<isize>);
make_conj!(Complex<i8>);
make_conj!(Complex<i16>);
make_conj!(Complex<i32>);
make_conj!(Complex<i64>);
make_conj!(Complex<f32>);
make_conj!(Complex<f64>);

// The tests expect the name for the crate containing the tested functions to be argmin_math
#[cfg(test)]
use crate as argmin_math;
include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/ndarray-tests-src/conj.rs"
));
