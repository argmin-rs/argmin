// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminMinMax;
use ndarray::{Array1, Array2};

macro_rules! make_minmax {
    ($t:ty) => {
        impl ArgminMinMax for Array1<$t> {
            #[inline]
            fn min(x: &Self, y: &Self) -> Array1<$t> {
                assert_eq!(x.shape(), y.shape());
                x.iter()
                    .zip(y)
                    .map(|(&a, &b)| if a < b { a } else { b })
                    .collect()
            }

            #[inline]
            fn max(x: &Self, y: &Self) -> Array1<$t> {
                assert_eq!(x.shape(), y.shape());
                x.iter()
                    .zip(y)
                    .map(|(&a, &b)| if a > b { a } else { b })
                    .collect()
            }
        }

        impl ArgminMinMax for Array2<$t> {
            #[inline]
            fn min(x: &Self, y: &Self) -> Array2<$t> {
                assert_eq!(x.shape(), y.shape());
                let m = x.shape()[0];
                let n = x.shape()[1];
                let mut out = x.clone();
                for i in 0..m {
                    for j in 0..n {
                        let a = x[(i, j)];
                        let b = y[(i, j)];
                        out[(i, j)] = if a < b { a } else { b };
                    }
                }
                out
            }

            #[inline]
            fn max(x: &Self, y: &Self) -> Array2<$t> {
                assert_eq!(x.shape(), y.shape());
                let m = x.shape()[0];
                let n = x.shape()[1];
                let mut out = x.clone();
                for i in 0..m {
                    for j in 0..n {
                        let a = x[(i, j)];
                        let b = y[(i, j)];
                        out[(i, j)] = if a > b { a } else { b };
                    }
                }
                out
            }
        }
    };
}

make_minmax!(isize);
make_minmax!(usize);
make_minmax!(i8);
make_minmax!(u8);
make_minmax!(i16);
make_minmax!(u16);
make_minmax!(i32);
make_minmax!(u32);
make_minmax!(i64);
make_minmax!(u64);
make_minmax!(f32);
make_minmax!(f64);

// All code that does not depend on a linked ndarray-linalg backend can still be tested as normal.
// To avoid dublicating tests and to allow convenient testing of functionality that does not need ndarray-linalg the tests are still included here.
// The tests expect the name for the crate containing the tested functions to be argmin_math
#[cfg(test)]
use crate as argmin_math;
include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/ndarray-tests-src/minmax.rs"
));
