// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::{ArgminTriangularSolve, Error};
use nalgebra::{DMatrix, DVector};
use std::fmt;

#[derive(Debug, thiserror::Error, PartialEq)]
struct SingularTriangularError;

impl fmt::Display for SingularTriangularError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Singular upper-triangular matrix")
    }
}

macro_rules! make_solve {
    ($t:ty) => {
        impl ArgminTriangularSolve<DVector<$t>, DVector<$t>> for DMatrix<$t> {
            #[inline]
            fn solve_upper_triangular(&self, b: &DVector<$t>) -> Result<DVector<$t>, Error> {
                match self.solve_upper_triangular(b) {
                    Some(x) => Ok(x),
                    None => Err(SingularTriangularError {}.into()),
                }
            }
        }

        impl ArgminTriangularSolve<DMatrix<$t>, DMatrix<$t>> for DMatrix<$t> {
            #[inline]
            fn solve_upper_triangular(&self, b: &DMatrix<$t>) -> Result<DMatrix<$t>, Error> {
                match self.solve_upper_triangular(b) {
                    Some(x) => Ok(x),
                    None => Err(SingularTriangularError {}.into()),
                }
            }
        }
    };
}

make_solve!(f32);
make_solve!(f64);

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_triangular_solve_ $t>]() {
                    let r = DMatrix::<$t>::from_row_slice(
                        3, 3,
                        &[
                            2.0, 1.0, 4.0,
                            0.0, 3.0, 5.0,
                            0.0, 0.0, 6.0,
                        ],
                    );
                    let b = DVector::<$t>::from_vec(vec![18.0, 33.0, 24.0]);
                    let x = ArgminTriangularSolve::solve_upper_triangular(&r, &b).unwrap();
                    let r_x = &r * &x;
                    for i in 0..3 {
                        assert_relative_eq!(r_x[i], b[i], epsilon = (1e-5) as $t);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_triangular_solve_singular_ $t>]() {
                    let r = DMatrix::<$t>::from_row_slice(
                        3, 3,
                        &[
                            2.0, 1.0, 4.0,
                            0.0, 0.0, 5.0,
                            0.0, 0.0, 6.0,
                        ],
                    );
                    let b = DVector::<$t>::from_vec(vec![1.0, 2.0, 3.0]);
                    let res: Result<DVector<$t>, Error> =
                        ArgminTriangularSolve::solve_upper_triangular(&r, &b);
                    assert!(res.is_err());
                }
            }
        };
    }

    make_test!(f32);
    make_test!(f64);
}
