// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminArgsort;
use ndarray::Array1;

macro_rules! make_argsort {
    ($t:ty) => {
        impl ArgminArgsort for Array1<$t> {
            #[inline]
            fn argsort(&self) -> Vec<usize> {
                let mut indices = (0..self.len()).collect::<Vec<_>>();
                indices.sort_by(|&i, &j| self[i].partial_cmp(&self[j]).unwrap());
                indices
            }
        }
    };
}

make_argsort!(i32);
make_argsort!(i64);
make_argsort!(f32);
make_argsort!(f64);

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_argsort() {
        assert_eq!(
            ArgminArgsort::argsort(&array![2, 4, 5, 2, 5, 0, 1]),
            vec![5, 6, 0, 3, 1, 2, 4]
        );
        assert_eq!(
            ArgminArgsort::argsort(&array![2., 4., 5., 2., 5., 0., 1.]),
            vec![5, 6, 0, 3, 1, 2, 4]
        );
    }
}
