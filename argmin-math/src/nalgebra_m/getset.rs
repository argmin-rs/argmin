// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::{ArgminGet, ArgminSet};
use nalgebra::{
    base::{
        dimension::Dim,
        storage::{Storage, StorageMut},
    },
    Matrix, Scalar,
};

impl<N, R, C, S> ArgminGet<usize, N> for Matrix<N, R, C, S>
where
    N: Copy + Scalar,
    R: Dim,
    C: Dim,
    S: Storage<N, R, C>,
{
    #[inline]
    fn get(&self, pos: usize) -> &N {
        &self[pos]
    }
}

impl<N, R, C, S> ArgminSet<usize, N> for Matrix<N, R, C, S>
where
    N: Copy + Scalar,
    R: Dim,
    C: Dim,
    S: StorageMut<N, R, C>,
{
    #[inline]
    fn set(&mut self, pos: usize, x: N) {
        self[pos] = x;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_get_set_ $t>]() {

                    let mut v = Vector3::zeros();
                    ArgminSet::<usize, $t>::set(&mut v, 0, 1 as $t);
                    ArgminSet::<usize, $t>::set(&mut v, 1, 2 as $t);
                    ArgminSet::<usize, $t>::set(&mut v, 2, 3 as $t);
                    assert_eq!(*ArgminGet::<usize, $t>::get(&v, 0) as i64, 1 as i64);
                    assert_eq!(*ArgminGet::<usize, $t>::get(&v, 1) as i64, 2 as i64);
                    assert_eq!(*ArgminGet::<usize, $t>::get(&v, 2) as i64, 3 as i64);

                }
            }
        };
    }

    make_test!(f32);
    make_test!(f64);
}
