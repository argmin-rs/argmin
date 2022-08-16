// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminRandomMatrix;
use ndarray::{Array, Array2};
use rand::Rng;
use rand_distr::StandardNormal;

macro_rules! make_random_normal {
    ($t:ty) => {
        impl ArgminRandomMatrix for Array2<$t> {
            #[inline]
            fn standard_normal(nrows: usize, ncols: usize) -> Array2<$t> {
                let mut rng = rand::thread_rng();
                Array::from_shape_fn((nrows, ncols), |_| rng.sample(StandardNormal))
            }
        }
    };
}

make_random_normal!(f32);
make_random_normal!(f64);

#[cfg(test)]
mod tests {
    use super::*;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_standard_normal_ $t>]() {
                    let e: Array2<$t> = <Array2<$t> as ArgminRandomMatrix>::standard_normal(100, 100);
                    let z_score = 1.96;
                    let sd = 1.0;

                    let n_outlier: usize = e.iter().map(|x: &$t| if *x > z_score * sd || *x < -(z_score * sd) {1} else {0}).sum();
                    assert!(n_outlier > 400 && n_outlier < 600); // Should be somewhere around 0.05 * 10000
                    assert_eq!(e.shape(), [100, 100]);
                }
            }
        };
    }

    make_test!(f32);
    make_test!(f64);
}
