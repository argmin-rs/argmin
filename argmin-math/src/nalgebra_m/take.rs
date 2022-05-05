// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminTake;
use nalgebra::{
    base::{dimension::Dim, DefaultAllocator, Scalar},
    OMatrix,
};

impl<N, R, C> ArgminTake<usize> for OMatrix<N, R, C>
where
    N: Copy + Scalar,
    R: Dim,
    C: Dim,
    DefaultAllocator: nalgebra::allocator::Allocator<N, R, C>,
{
    #[inline]
    fn take(&self, indices: &[usize], axis: u8) -> Self {
        match axis {
            0 => Self::from_iterator_generic(
                R::from_usize(indices.len()),
                C::from_usize(self.ncols()),
                (0..self.ncols()).flat_map(|i| indices.iter().map(move |&j| self[(j, i)])),
            ),
            _ => Self::from_iterator_generic(
                R::from_usize(self.nrows()),
                C::from_usize(indices.len()),
                indices
                    .iter()
                    .flat_map(|&i| (0..self.nrows()).map(move |j| self[(j, i)])),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{dmatrix, dvector, matrix, vector};

    #[test]
    fn test_take() {
        let m = dmatrix![1, 2, 3; 4, 5, 6; 7, 8, 9];
        let v1 = dvector![2, 4, 5, 2, 5, 0, 1];
        let v2 = vector![1, 2, 1];

        assert_eq!(ArgminTake::take(&v1, &[5usize, 6, 0], 0), vector![0, 1, 2]);
        assert_eq!(ArgminTake::take(&v2, &[0usize, 0, 0], 0), vector![1, 1, 1]);

        assert_eq!(
            ArgminTake::take(&m, &[2usize, 0], 0),
            dmatrix![7, 8, 9; 1, 2, 3]
        );
        assert_eq!(
            ArgminTake::take(&m, &[0usize, 0, 0, 0], 0),
            matrix![1, 2, 3; 1, 2, 3; 1, 2, 3; 1, 2, 3]
        );
        assert_eq!(
            ArgminTake::take(&m, &[2usize, 1], 1),
            matrix![3, 2; 6, 5; 9, 8]
        );
    }
}
