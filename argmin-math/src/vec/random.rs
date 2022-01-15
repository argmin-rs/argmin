// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminRandom;
use rand::distributions::uniform::SampleUniform;
use rand::Rng;

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
