// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::{ArgminAxisIter, ArgminIter, ArgminMutIter};
use nalgebra::{
    base::{
        allocator::Allocator,
        dimension::Dim,
        storage::{Storage, StorageMut},
        Const, U1,
    },
    DefaultAllocator, Matrix, OMatrix, Scalar,
};

impl<N, R, C, S> ArgminIter<N> for Matrix<N, R, C, S>
where
    R: Dim,
    C: Dim,
    S: Storage<N, R, C>,
{
    #[inline]
    fn iterator<'a>(&'a self) -> Box<dyn Iterator<Item = &'a N> + 'a> {
        Box::new(self.iter())
    }
}

impl<N, R, C, S> ArgminMutIter<N> for Matrix<N, R, C, S>
where
    R: Dim,
    C: Dim,
    S: StorageMut<N, R, C>,
{
    #[inline]
    fn iterator_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut N> + 'a>
    where
        S:,
    {
        Box::new(self.iter_mut())
    }
}

impl<N, R, C, O, S> ArgminAxisIter<OMatrix<N, O, U1>> for Matrix<N, R, C, S>
where
    N: Copy + Scalar,
    R: Dim,
    C: Dim,
    O: Dim,
    S: Storage<N, R, C>,
    DefaultAllocator: Allocator<N, O, U1>,
{
    #[inline]
    fn row_iterator<'a>(&'a self) -> Box<dyn Iterator<Item = OMatrix<N, O, U1>> + 'a> {
        Box::new(self.row_iter().map(|vec| {
            OMatrix::<N, O, U1>::from_iterator_generic(
                O::from_usize(self.ncols()),
                Const::<1>,
                vec.iter().copied(),
            )
        }))
    }

    #[inline]
    fn column_iterator<'a>(&'a self) -> Box<dyn Iterator<Item = OMatrix<N, O, U1>> + 'a> {
        Box::new(self.column_iter().map(|vec| {
            OMatrix::<N, O, U1>::from_iterator_generic(
                O::from_usize(self.nrows()),
                Const::<1>,
                vec.iter().copied(),
            )
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{matrix, vector, Vector2, Vector3};

    #[test]
    fn test_iter() {
        assert_eq!(
            ArgminIter::iterator(&vector![1i32, 2, 3])
                .map(|&e| e * 2)
                .collect::<Vec<i32>>(),
            vec![2i32, 4, 6]
        );
        let mut v = vector![1i32, 2, 3];
        ArgminMutIter::iterator_mut(&mut v).for_each(|e| *e *= 2);
        assert_eq!(v, vector![2i32, 4, 6]);
    }

    #[test]
    fn test_axis_iter() {
        let m = matrix![1, 2, 3; 4, 5, 6];

        let row_sums: Vec<i32> = m.row_iterator().map(|v: Vector3<i32>| v.sum()).collect();
        let col_sums: Vec<i32> = m.column_iterator().map(|v: Vector2<i32>| v.sum()).collect();

        assert_eq!(row_sums, vec![6, 15]);
        assert_eq!(col_sums, vec![5, 7, 9]);
    }
}
