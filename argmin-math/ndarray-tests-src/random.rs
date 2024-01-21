// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;
    use argmin_math::ArgminRandom;
    use ndarray::{array, Array1, Array2};
    use paste::item;
    use rand::SeedableRng;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_random_vec_ $t>]() {
                    let a = array![1 as $t, 2 as $t, 4 as $t];
                    let b = array![2 as $t, 3 as $t, 5 as $t];
                    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
                    let random = Array1::<$t>::rand_from_range(&a, &b, &mut rng);
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
                    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
                    let random = Array2::<$t>::rand_from_range(&a, &b, &mut rng);
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
