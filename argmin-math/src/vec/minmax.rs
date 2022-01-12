// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminMinMax;

impl<T> ArgminMinMax for Vec<T>
where
    T: std::cmp::PartialOrd + Clone,
{
    fn min(x: &Self, y: &Self) -> Vec<T> {
        assert!(!x.is_empty());
        assert_eq!(x.len(), y.len());

        x.iter()
            .zip(y.iter())
            .map(|(a, b)| if a < b { a.clone() } else { b.clone() })
            .collect()
    }

    fn max(x: &Self, y: &Self) -> Vec<T> {
        assert!(!x.is_empty());
        assert_eq!(x.len(), y.len());

        x.iter()
            .zip(y.iter())
            .map(|(a, b)| if a > b { a.clone() } else { b.clone() })
            .collect()
    }
}
