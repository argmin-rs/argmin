// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminOuter;
use nalgebra::{
    base::{allocator::Allocator, dimension::Dim, storage::Storage, U1},
    constraint::{AreMultipliable, ShapeConstraint},
    ClosedAdd, ClosedMul, DefaultAllocator, OMatrix, OVector, Scalar, Vector,
};
use num_traits::{One, Zero};

impl<N, R, C, S> ArgminOuter<Vector<N, C, S>, OMatrix<N, R, C>> for OVector<N, R>
where
    N: Scalar + Zero + One + ClosedAdd + ClosedMul,
    R: Dim,
    C: Dim,
    S: Storage<N, C>,
    DefaultAllocator: Allocator<N, U1, C>,
    DefaultAllocator: Allocator<N, R, C>,
    DefaultAllocator: Allocator<N, C>,
    DefaultAllocator: Allocator<N, R>,
    ShapeConstraint: AreMultipliable<R, U1, U1, C>,
{
    #[inline]
    fn outer(&self, other: &Vector<N, C, S>) -> OMatrix<N, R, C> {
        self * other.transpose()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{matrix, vector, Vector3, Vector4};

    #[test]
    fn test_outer() {
        let v1: Vector3<i32> = vector![1, 2, 3];
        let v2: Vector4<i32> = vector![4, 5, 6, 7];

        let expected = matrix![4, 5, 6, 7; 8, 10, 12, 14; 12, 15, 18, 21];

        let product = v1.outer(&v2);

        assert_eq!(product.shape(), (3usize, 4usize));

        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(expected[(i, j)], product[(i, j)]);
            }
        }
    }
}
