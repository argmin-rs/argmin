// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::{ArgminInv, Error};
use nalgebra::{
    base::{allocator::Allocator, dimension::Dim, storage::Storage},
    ComplexField, DefaultAllocator, OMatrix, SquareMatrix,
};
use std::fmt;

#[derive(Debug, thiserror::Error, PartialEq)]
struct InverseError;

impl fmt::Display for InverseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Non-invertible matrix")
    }
}

impl<N, D, S> ArgminInv<OMatrix<N, D, D>> for SquareMatrix<N, D, S>
where
    N: ComplexField,
    D: Dim,
    S: Storage<N, D, D>,
    DefaultAllocator: Allocator<N, D, D>,
{
    #[inline]
    fn inv(&self) -> Result<OMatrix<N, D, D>, Error> {
        match self.clone_owned().try_inverse() {
            Some(m) => Ok(m),
            None => Err(InverseError {}.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
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
                            assert_relative_eq!(res[(i, j)], target[(i, j)], epsilon = $t::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_inv_error $t>]() {
                    let a = Matrix2::new(
                        2 as $t, 5 as $t,
                        4 as $t, 10 as $t,
                    );
                    let err = <Matrix2<$t> as ArgminInv<Matrix2<$t>>>::inv(&a).unwrap_err().downcast::<InverseError>().unwrap();
                    assert_eq!(err, InverseError {});
                    assert_eq!(format!("{}", err), "Non-invertible matrix");
                    assert_eq!(format!("{:?}", err), "InverseError");
                }
            }
        };
    }

    make_test!(f32);
    make_test!(f64);
}
