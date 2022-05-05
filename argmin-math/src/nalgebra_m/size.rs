// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

// Note: This is not really the preferred way I think. Maybe this should also be implemented for
// ArrayViews, which would probably make it more efficient.

use crate::ArgminSize;
use nalgebra::{
    base::{allocator::Allocator, dimension::Dim, storage::Storage, Scalar},
    DefaultAllocator, Matrix,
};

impl<N, R, C, S> ArgminSize<usize> for Matrix<N, R, C, S>
where
    N: Scalar,
    R: Dim,
    C: Dim,
    S: Storage<N, R, C>,
    DefaultAllocator: Allocator<N, C, R>,
{
    #[inline]
    fn shape(&self) -> usize {
        self.len()
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use nalgebra::{DVector, Vector2};

    #[test]
    fn test_transitions() {
        let static_vec = Vector2::new(1, 4);
        let dynamic_vec = DVector::from_vec(vec![1, 2, 3, 4, 5]);

        assert_eq!(2, ArgminSize::<usize>::shape(&static_vec));
        assert_eq!(5, ArgminSize::<usize>::shape(&dynamic_vec));
    }
}
