// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminTake;
use ndarray::{Array, Array1, Array2};

macro_rules! make_take {
    ($t:ty) => {
        impl ArgminTake<usize> for Array1<$t> {
            #[inline]
            fn take(&self, indices: &[usize], _: u8) -> Self {
                indices.iter().map(|&i| self[i]).collect()
            }
        }

        impl ArgminTake<usize> for Array2<$t> {
            #[inline]
            fn take(&self, indices: &[usize], axis: u8) -> Self {
                match axis {
                    0 => Array::from_iter(
                        indices
                            .iter()
                            .flat_map(|&i| (0..self.ncols()).map(move |j| self[[i, j]])),
                    )
                    .into_shape((indices.len(), self.ncols()))
                    .unwrap(),
                    1 => Array::from_iter(
                        (0..self.nrows()).flat_map(|j| indices.iter().map(move |&i| self[[j, i]])),
                    )
                    .into_shape((self.nrows(), indices.len()))
                    .unwrap(),
                    _ => panic!("Axis value {} should be either 0 or 1", axis),
                }
            }
        }
    };
}

make_take!(isize);
make_take!(usize);
make_take!(i8);
make_take!(i16);
make_take!(i32);
make_take!(i64);
make_take!(u8);
make_take!(u16);
make_take!(u32);
make_take!(u64);
make_take!(f32);
make_take!(f64);

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_take() {
        let m: Array2<i32> = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        let v1: Array1<i32> = array![2, 4, 5, 2, 5, 0, 1];
        let v2: Array1<i32> = array![1, 2];

        assert_eq!(ArgminTake::take(&v1, &[5usize, 6, 0], 0), array![0, 1, 2]);
        assert_eq!(
            ArgminTake::take(&v2, &[0usize, 0, 0, 0], 0),
            array![1, 1, 1, 1]
        );

        assert_eq!(
            ArgminTake::take(&m, &[2usize, 0], 0),
            array![[7, 8, 9], [1, 2, 3],]
        );
        assert_eq!(
            ArgminTake::take(&m, &[0usize, 0, 0, 0], 0),
            array![[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3],]
        );
        assert_eq!(
            ArgminTake::take(&m, &[2usize, 1], 1),
            array![[3, 2], [6, 5], [9, 8],]
        );
    }
}
