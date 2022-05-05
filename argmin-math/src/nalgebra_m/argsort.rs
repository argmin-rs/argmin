// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminArgsort;
use nalgebra::{
    base::{dimension::Dim, storage::Storage},
    Matrix,
};

impl<N, R, C, S> ArgminArgsort for Matrix<N, R, C, S>
where
    N: PartialOrd,
    R: Dim,
    C: Dim,
    S: Storage<N, R, C>,
{
    #[inline]
    fn argsort(&self) -> Vec<usize> {
        let (n, m) = self.shape();
        let l = if n > m { n } else { m };
        let mut indices = (0..l).collect::<Vec<_>>();
        indices.sort_by(|&i, &j| self[i].partial_cmp(&self[j]).unwrap());
        indices
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::matrix;

    #[test]
    fn test_argsort() {
        assert_eq!(
            ArgminArgsort::argsort(&matrix![2, 4, 5, 2, 5, 0, 1]),
            vec![5, 6, 0, 3, 1, 2, 4]
        );
        assert_eq!(
            ArgminArgsort::argsort(&matrix![2., 4., 5., 2., 5., 0., 1.]),
            vec![5, 6, 0, 3, 1, 2, 4]
        );
    }
}
