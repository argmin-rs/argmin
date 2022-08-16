// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

// Note: This is not really the preferred way I think. Maybe this should also be implemented for
// ArrayViews, which would probably make it more efficient.

use crate::ArgminTransition;
use nalgebra::{
    base::{dimension::Dim, storage::Storage, Scalar},
    DMatrix, DVector, Matrix,
};

impl<N, R, C, S> ArgminTransition for Matrix<N, R, C, S>
where
    N: Scalar,
    R: Dim,
    C: Dim,
    S: Storage<N, R, C>,
{
    type Array1D = DVector<N>;
    type Array2D = DMatrix<N>;
}

#[cfg(test)]
mod tests {
    use crate::ArgminEye;

    use super::*;
    use nalgebra::{DVector, Vector2};

    #[test]
    fn test_transitions() {
        let static_vec = Vector2::new(1, 4);
        let dynamic_vec = DVector::from_vec(vec![1, 2, 3, 4, 5]);

        assert_eq!(DMatrix::<i32>::eye(3), eye_from_vec(&static_vec, 3));
        assert_eq!(DMatrix::<i32>::eye(3), eye_from_vec(&dynamic_vec, 3));
    }

    fn eye_from_vec<V: ArgminTransition>(_: &V, size: usize) -> V::Array2D
    where
        V::Array2D: ArgminEye,
    {
        V::Array2D::eye(size)
    }
}
