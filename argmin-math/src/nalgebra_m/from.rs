// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminFrom;
use nalgebra::{
    base::{allocator::Allocator, dimension::Dim},
    DefaultAllocator, OMatrix, Scalar,
};

impl<'a, N, R, C> ArgminFrom<&'a N, usize> for OMatrix<N, R, C>
where
    N: Copy + Scalar,
    R: Dim,
    C: Dim,
    DefaultAllocator: Allocator<N, R, C>,
{
    #[inline]
    fn from_iterator<I: Iterator<Item = &'a N>>(len: usize, iter: I) -> Self {
        OMatrix::<N, R, C>::from_iterator_generic(
            R::from_usize(len),
            C::from_usize(1),
            iter.take(len).copied(),
        )
    }
}

impl<N, R, C> ArgminFrom<N, usize> for OMatrix<N, R, C>
where
    N: Copy + Scalar,
    R: Dim,
    C: Dim,
    DefaultAllocator: Allocator<N, R, C>,
{
    #[inline]
    fn from_iterator<I: Iterator<Item = N>>(len: usize, iter: I) -> Self {
        OMatrix::<N, R, C>::from_iterator_generic(
            R::from_usize(len),
            C::from_usize(1),
            iter.take(len),
        )
    }
}

impl<N, R, C> ArgminFrom<N, (usize, usize)> for OMatrix<N, R, C>
where
    N: Copy + Scalar,
    R: Dim,
    C: Dim,
    DefaultAllocator: Allocator<N, R, C>,
    DefaultAllocator: Allocator<N, C, R>,
{
    #[inline]
    fn from_iterator<I: Iterator<Item = N>>(shape: (usize, usize), iter: I) -> Self {
        let mut iter = iter;
        OMatrix::<N, C, R>::from_fn_generic(
            C::from_usize(shape.1),
            R::from_usize(shape.0),
            |_, _| iter.next().unwrap(),
        )
        .transpose()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{matrix, vector, DVector, Matrix3x2, Vector3};
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_from_array_ $t>]() {

                    let v: Vector3<$t> = ArgminFrom::<&$t, usize>::from_iterator(3usize, [1 as $t, 2 as $t, 3 as $t].iter());
                    assert_eq!(v, vector![1 as $t, 2 as $t, 3 as $t]);

                }

                #[test]
                fn [<test_from_range_ $t>]() {

                    let v: Vector3<$t> = ArgminFrom::<$t, usize>::from_iterator(3usize, (1..4).map(|v| v as $t));
                    assert_eq!(v, vector![1 as $t, 2 as $t, 3 as $t]);

                }

                #[test]
                fn [<test_from_dvector_ $t>]() {

                    let v: DVector<$t> = ArgminFrom::<&$t, usize>::from_iterator(3usize, [1 as $t, 2 as $t, 3 as $t].iter());
                    assert_eq!(v, vector![1 as $t, 2 as $t, 3 as $t]);

                }

                #[test]
                fn [<test_from_range_2_ $t>]() {

                    let m: Matrix3x2<$t> = ArgminFrom::<$t, (usize, usize)>::from_iterator((3usize, 2usize), (0..6).map(|v| v as $t));
                    assert_eq!(m, matrix![0 as $t, 1 as $t; 2 as $t, 3 as $t; 4 as $t, 5 as $t]);

                }

                #[test]
                fn [<test_from_darray_ $t>]() {

                    let m: Matrix3x2<$t> = ArgminFrom::<$t, (usize, usize)>::from_iterator((3usize, 2usize), (0..6).map(|v| v as $t));
                    assert_eq!(m, matrix![0 as $t, 1 as $t; 2 as $t, 3 as $t; 4 as $t, 5 as $t]);

                }
            }

        };
    }

    make_test!(f32);
    make_test!(f64);
}
