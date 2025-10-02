// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use rand::Rng;

use crate::ArgminRandom;

macro_rules! make_random {
    ($t:ty) => {
        impl ArgminRandom for ndarray::Array1<$t> {
            fn rand_from_range<R: Rng>(min: &Self, max: &Self, rng: &mut R) -> ndarray::Array1<$t> {
                assert!(!min.is_empty());
                assert_eq!(min.len(), max.len());

                ndarray::Array1::from_iter(min.iter().zip(max.iter()).map(|(a, b)| {
                    // Do not require a < b:

                    // We do want to know if a and b are *exactly* the same.
                    #[allow(clippy::float_cmp)]
                    if a == b {
                        a.clone()
                    } else if a < b {
                        rng.random_range(a.clone()..b.clone())
                    } else {
                        rng.random_range(b.clone()..a.clone())
                    }
                }))
            }
        }

        impl ArgminRandom for ndarray::Array2<$t> {
            fn rand_from_range<R: Rng>(min: &Self, max: &Self, rng: &mut R) -> ndarray::Array2<$t> {
                assert!(!min.is_empty());
                assert_eq!(min.raw_dim(), max.raw_dim());

                ndarray::Array2::from_shape_fn(min.raw_dim(), |(i, j)| {
                    let a = min.get((i, j)).unwrap();
                    let b = max.get((i, j)).unwrap();

                    // We do want to know if a and b are *exactly* the same.
                    #[allow(clippy::float_cmp)]
                    if a == b {
                        a.clone()
                    } else if a < b {
                        rng.random_range(a.clone()..b.clone())
                    } else {
                        rng.random_range(b.clone()..a.clone())
                    }
                })
            }
        }
    };
}

make_random!(i8);
make_random!(u8);
make_random!(i16);
make_random!(u16);
make_random!(i32);
make_random!(u32);
make_random!(i64);
make_random!(u64);
make_random!(f32);
make_random!(f64);

// All code that does not depend on a linked ndarray-linalg backend can still be tested as normal.
// To avoid duplicating tests and to allow convenient testing of functionality that does not need ndarray-linalg the tests are still included here.
// The tests expect the name for the crate containing the tested functions to be argmin_math
#[cfg(test)]
use crate as argmin_math;
include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/ndarray-tests-src/random.rs"
));
