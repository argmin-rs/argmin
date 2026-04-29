// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::{ArgminTriangularSolve, Error};
use faer::{linalg::triangular_solve::solve_upper_triangular_in_place, Mat, Par};
use faer_traits::{math_utils, ComplexField, RealField};
use std::fmt;

#[derive(Debug, thiserror::Error, PartialEq)]
struct SingularTriangularError;

impl fmt::Display for SingularTriangularError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Singular upper-triangular matrix")
    }
}

impl<E: RealField + PartialOrd> ArgminTriangularSolve<Mat<E>, Mat<E>> for Mat<E> {
    #[inline]
    fn solve_upper_triangular(&self, b: &Mat<E>) -> Result<Mat<E>, Error> {
        assert_eq!(self.nrows(), self.ncols(), "R must be square");
        let n = self.nrows();
        // Detect singular (any zero on the diagonal).
        for i in 0..n {
            let d = &self[(i, i)];
            if !math_utils::is_finite(d) || math_utils::abs(d) <= math_utils::min_positive::<E>() {
                return Err(SingularTriangularError {}.into());
            }
        }
        let mut x = b.clone();
        solve_upper_triangular_in_place(self.as_ref(), x.as_mut(), Par::Seq);
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use faer::mat;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_triangular_solve_ $t>]() {
                    let r = mat![
                        [2.0 as $t, 1.0, 4.0],
                        [0.0, 3.0, 5.0],
                        [0.0, 0.0, 6.0]
                    ];
                    let b = mat![[18.0 as $t], [33.0], [24.0]];
                    let x = ArgminTriangularSolve::solve_upper_triangular(&r, &b).unwrap();
                    let r_x = &r * &x;
                    for i in 0..3 {
                        assert_relative_eq!(r_x[(i, 0)], b[(i, 0)], epsilon = 1e-5 as $t);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_triangular_solve_singular_ $t>]() {
                    let r = mat![
                        [2.0 as $t, 1.0, 4.0],
                        [0.0, 0.0, 5.0],
                        [0.0, 0.0, 6.0]
                    ];
                    let b = mat![[1.0 as $t], [2.0], [3.0]];
                    let res: Result<Mat<$t>, Error> =
                        ArgminTriangularSolve::solve_upper_triangular(&r, &b);
                    assert!(res.is_err());
                }
            }
        };
    }

    make_test!(f32);
    make_test!(f64);
}
