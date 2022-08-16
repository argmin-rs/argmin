// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminOuter;
use ndarray::{Array1, Array2};

macro_rules! make_eigen {
    ($t:ty) => {
        impl ArgminOuter<Array1<$t>, Array2<$t>> for Array1<$t> {
            #[inline]
            fn outer(&self, other: &Array1<$t>) -> Array2<$t> {
                let (size_x, size_y) = (self.len(), other.len());
                let x_reshaped = self.view().into_shape((size_x, 1)).unwrap();
                let y_reshaped = other.view().into_shape((1, size_y)).unwrap();
                x_reshaped.dot(&y_reshaped)
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
    use ndarray::{array, Array2};

    #[test]
    fn test_outer() {
        let v1 = array![1, 2, 3];
        let v2 = array![4, 5, 6];

        let expected = array![[4, 5, 6], [8, 10, 12], [12, 15, 18]];

        let product: Array2<i32> = v1.outer(&v2);

        assert_eq!(product.shape(), [3usize, 3usize]);

        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(expected[[i, j]], product[[i, j]]);
            }
        }
    }
}
