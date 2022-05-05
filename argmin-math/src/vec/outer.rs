// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminOuter;

macro_rules! make_eigen {
    ($t:ty) => {
        impl ArgminOuter<Vec<$t>, Vec<Vec<$t>>> for Vec<$t> {
            #[inline]
            fn outer(&self, other: &Vec<$t>) -> Vec<Vec<$t>> {
                let (n, m) = (self.len(), other.len());
                (0..n)
                    .map(|i| (0..m).map(|j| self[i] * other[j]).collect())
                    .collect()
            }
        }
    };
}

make_eigen!(i32);
make_eigen!(i64);
make_eigen!(f32);
make_eigen!(f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argsort() {
        let v1 = vec![1, 2, 3];
        let v2 = vec![4, 5, 6];

        let expected = vec![4, 5, 6, 8, 10, 12, 12, 15, 18];

        let product: Vec<Vec<i32>> = v1.outer(&v2);

        assert_eq!(product.len(), 3);
        assert_eq!(product[0].len(), 3);

        product
            .iter()
            .flatten()
            .zip(expected)
            .for_each(|(&v1, v2)| assert_eq!(v1, v2));
    }
}
