// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminTake;

macro_rules! make_take {
    ($t:ty) => {
        impl ArgminTake<usize> for Vec<$t> {
            #[inline]
            fn take(&self, indices: &[usize], _: u8) -> Self {
                indices.iter().map(|&i| self[i]).collect()
            }
        }

        impl ArgminTake<usize> for Vec<Vec<$t>> {
            #[inline]
            fn take(&self, indices: &[usize], axis: u8) -> Self {
                match axis {
                    0 => indices.iter().map(|&i| self[i].clone()).collect(),
                    1 => (0..self.len())
                        .map(|i| indices.iter().map(|&j| self[i][j]).collect())
                        .collect(),
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

    #[test]
    fn test_take() {
        let m = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];

        assert_eq!(
            ArgminTake::take(&vec![2, 4, 5, 2, 5, 0, 1], &[5usize, 6, 0], 0),
            vec![0, 1, 2]
        );
        assert_eq!(
            ArgminTake::take(&vec![1, 2], &[0usize, 0, 0, 0], 0),
            vec![1, 1, 1, 1]
        );

        assert_eq!(
            ArgminTake::take(&m, &[2usize, 0], 0),
            vec![vec![7, 8, 9], vec![1, 2, 3],]
        );
        assert_eq!(
            ArgminTake::take(&m, &[0usize, 0, 0, 0], 0),
            vec![vec![1, 2, 3], vec![1, 2, 3], vec![1, 2, 3], vec![1, 2, 3],]
        );
        assert_eq!(
            ArgminTake::take(&m, &[2usize, 1], 1),
            vec![vec![3, 2], vec![6, 5], vec![9, 8],]
        );
    }
}
