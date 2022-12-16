// Copyright 2018-2022 argmin developers
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
            fn rand_from_range(min: &Self, max: &Self) -> ndarray::Array1<$t> {
                assert!(!min.is_empty());
                assert_eq!(min.len(), max.len());

                let mut rng = rand::thread_rng();

                ndarray::Array1::from_iter(min.iter().zip(max.iter()).map(|(a, b)| {
                    // Do not require a < b:

                    // We do want to know if a and b are *exactly* the same.
                    #[allow(clippy::float_cmp)]
                    if a == b {
                        a.clone()
                    } else if a < b {
                        rng.gen_range(a.clone()..b.clone())
                    } else {
                        rng.gen_range(b.clone()..a.clone())
                    }
                }))
            }
        }

        impl ArgminRandom for ndarray::Array2<$t> {
            fn rand_from_range(min: &Self, max: &Self) -> ndarray::Array2<$t> {
                assert!(!min.is_empty());
                assert_eq!(min.raw_dim(), max.raw_dim());

                let mut rng = rand::thread_rng();

                ndarray::Array2::from_shape_fn(min.raw_dim(), |(i, j)| {
                    let a = min.get((i, j)).unwrap();
                    let b = max.get((i, j)).unwrap();

                    // We do want to know if a and b are *exactly* the same.
                    #[allow(clippy::float_cmp)]
                    if a == b {
                        a.clone()
                    } else if a < b {
                        rng.gen_range(a.clone()..b.clone())
                    } else {
                        rng.gen_range(b.clone()..a.clone())
                    }
                })
            }
        }
    };
}

make_random!(isize);
make_random!(usize);
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, Array2};
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_random_vec_ $t>]() {
                    let a = array![1 as $t, 2 as $t, 4 as $t];
                    let b = array![2 as $t, 3 as $t, 5 as $t];
                    let random = Array1::<$t>::rand_from_range(&a, &b);
                    for i in 0..3usize {
                        assert!(random[i] >= a[i]);
                        assert!(random[i] <= b[i]);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_random_mat_ $t>]() {
                    let a = array![
                        [1 as $t, 2 as $t, 4 as $t],
                        [2 as $t, 3 as $t, 5 as $t]
                    ];
                    let b = array![
                        [2 as $t, 3 as $t, 5 as $t],
                        [3 as $t, 4 as $t, 6 as $t]
                    ];
                    let random = Array2::<$t>::rand_from_range(&a, &b);
                    for i in 0..3 {
                        for j in 0..2 {
                            assert!(random[(j, i)] >= a[(j, i)]);
                            assert!(random[(j, i)] <= b[(j, i)]);
                        }
                    }
                }
            }
        };
    }

    make_test!(i8);
    make_test!(u8);
    make_test!(i16);
    make_test!(u16);
    make_test!(i32);
    make_test!(u32);
    make_test!(i64);
    make_test!(u64);
    make_test!(isize);
    make_test!(usize);
    make_test!(f32);
    make_test!(f64);
}
