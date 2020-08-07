// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::{errors::ArgminError, math::ArgminInv, Error};

use nalgebra::{
    base::{allocator::Allocator, dimension::Dim, storage::Storage},
    ComplexField, DefaultAllocator, MatrixN, SquareMatrix,
};

impl<N, D, S> ArgminInv<MatrixN<N, D>> for SquareMatrix<N, D, S>
where
    N: ComplexField,
    D: Dim,
    S: Storage<N, D, D>,
    DefaultAllocator: Allocator<N, D, D>,
{
    #[inline]
    fn inv(&self) -> Result<MatrixN<N, D>, Error> {
        match self.clone_owned().try_inverse() {
            Some(m) => Ok(m),
            None => Err(ArgminError::InvalidParameter {
                text: "ArgminInv: non-invertible matrix".to_string(),
            }
            .into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Matrix2;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_inv_ $t>]() {
                    let a = Matrix2::new(
                        2 as $t, 5 as $t,
                        1 as $t, 3 as $t,
                    );
                    let target = Matrix2::new(
                        3 as $t, -5 as $t,
                        -1 as $t, 2 as $t,
                    );
                    let res = <Matrix2<$t> as ArgminInv<Matrix2<$t>>>::inv(&a).unwrap();
                    for i in 0..2 {
                        for j in 0..2 {
                            assert!((((res[(i, j)] - target[(i, j)]) as f64).abs()) < std::f64::EPSILON);
                        }
                    }
                }
            }
        };
    }

    make_test!(f32);
    make_test!(f64);
}
