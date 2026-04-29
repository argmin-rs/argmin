// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::{ArgminPivotedQR, Error, PivotedQR};
use faer::{linalg::solvers::ColPivQr, perm::Perm, Mat, RealField, SimpleEntity};

impl<E: SimpleEntity + RealField> ArgminPivotedQR for Mat<E> {
    type Q = Mat<E>;
    type R = Mat<E>;
    type Permutation = Perm<usize>;
    type ColumnNorms = Mat<E>;

    #[inline]
    fn qr_pivot(
        self,
    ) -> Result<PivotedQR<Self::Q, Self::R, Self::Permutation, Self::ColumnNorms>, Error> {
        let n = self.ncols();
        // Column L2 norms in the original (un-pivoted) order.
        let mut column_norms = Mat::<E>::zeros(n, 1);
        for j in 0..n {
            column_norms[(j, 0)] = self.col_as_slice(j).l2_norm();
        }
        let qr = ColPivQr::new(self.as_ref());
        let q = qr.compute_thin_q();
        let r = qr.compute_thin_r();
        let perm_ref = qr.col_permutation();
        let (forward, inverse) = perm_ref.arrays();
        let permutation = unsafe {
            Perm::new_unchecked(
                forward.to_vec().into_boxed_slice(),
                inverse.to_vec().into_boxed_slice(),
            )
        };
        Ok(PivotedQR {
            q,
            r,
            permutation,
            column_norms,
        })
    }
}

trait L2NormSlice<E> {
    fn l2_norm(&self) -> E;
}

impl<E: SimpleEntity + RealField> L2NormSlice<E> for [E] {
    #[inline]
    fn l2_norm(&self) -> E {
        let mut sumsq = E::faer_zero();
        for x in self {
            sumsq = sumsq.faer_add(x.faer_mul(*x));
        }
        sumsq.faer_sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ArgminPermute;
    use approx::assert_relative_eq;
    use faer::mat;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_qr_pivot_reconstruct_ $t>]() {
                    let j = mat![
                        [1.0 as $t, 0.1, 5.0],
                        [2.0 as $t, 0.2, 6.0],
                        [3.0 as $t, 0.3, 7.0],
                        [4.0 as $t, 0.4, 8.0]
                    ];
                    let pqr = j.clone().qr_pivot().unwrap();

                    // J = Q · R · Pᵀ. Q*R has columns in pivoted order;
                    // `permute(row)` rebuilds the row in the original order
                    // (`permute(v)[k] = v[inverse[k]]`).
                    let qr = &pqr.q * &pqr.r;
                    for i in 0..4 {
                        let mut row_vec = Mat::<$t>::zeros(3, 1);
                        for k in 0..3 {
                            row_vec[(k, 0)] = qr[(i, k)];
                        }
                        let recovered = pqr.permutation.permute(&row_vec);
                        for k in 0..3 {
                            assert_relative_eq!(recovered[(k, 0)], j[(i, k)], epsilon = 1e-5 as $t);
                        }
                    }

                    // R is upper triangular.
                    for i in 1..3 {
                        for k in 0..i {
                            assert_relative_eq!(pqr.r[(i, k)], 0 as $t, epsilon = 1e-5 as $t);
                        }
                    }

                    // column_norms holds the L2 norms of the original columns.
                    for k in 0..3 {
                        let mut sumsq = 0 as $t;
                        for i in 0..4 {
                            sumsq += j[(i, k)] * j[(i, k)];
                        }
                        let expected = sumsq.sqrt();
                        assert_relative_eq!(pqr.column_norms[(k, 0)], expected, epsilon = 1e-5 as $t);
                    }
                }
            }
        };
    }

    make_test!(f32);
    make_test!(f64);
}
