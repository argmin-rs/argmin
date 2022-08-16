// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::{ArgminRandom, ArgminRandomMatrix};
use rand::distributions::uniform::SampleUniform;
use rand::Rng;
use rand_distr::StandardNormal;

impl<T> ArgminRandom for Vec<T>
where
    T: SampleUniform + std::cmp::PartialOrd + Clone,
{
    fn rand_from_range(min: &Self, max: &Self) -> Vec<T> {
        assert!(!min.is_empty());
        assert_eq!(min.len(), max.len());

        let mut rng = rand::thread_rng();

        min.iter()
            .zip(max.iter())
            .map(|(a, b)| {
                // Do not require a < b:

                if a == b {
                    a.clone()
                } else if a < b {
                    rng.gen_range(a.clone()..b.clone())
                } else {
                    rng.gen_range(b.clone()..a.clone())
                }
            })
            .collect()
    }
}

macro_rules! make_random_normal {
    ($t:ty) => {
        impl ArgminRandomMatrix for Vec<Vec<$t>> {
            #[inline]
            fn standard_normal(nrows: usize, ncols: usize) -> Vec<Vec<$t>> {
                let mut rng = rand::thread_rng();
                let mut out = vec![vec![0 as $t; ncols]; nrows];
                for i in 0..nrows {
                    for j in 0..ncols {
                        out[i][j] = rng.sample(StandardNormal);
                    }
                }
                out
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
                    let e: Vec<Vec<$t>> = <Vec<Vec<$t>> as ArgminRandomMatrix>::standard_normal(10,1000);
                    let z_score = 1.96;
                    let sd = 1.0;

                    let n_outlier: usize = e.iter().flat_map(|x: &Vec<_>| x.iter().map(|x: &$t| if *x > z_score * sd || *x < -(z_score * sd) {1} else {0})).sum();
                    assert!(n_outlier > 400 && n_outlier < 600); // Should be somewhere around 0.05 * 10000
                    assert_eq!(e.len(), 10);
                    assert_eq!(e[0].len(), 1000);
                }
            }

        };
    }

    make_test!(f32);
    make_test!(f64);
}
