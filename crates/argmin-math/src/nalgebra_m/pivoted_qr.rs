// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use super::Dyn;
use crate::{ArgminPivotedQR, Error, PivotedQR};
use nalgebra::{DMatrix, DVector, PermutationSequence};

// Implemented for `DMatrix<f32 | f64>` only. The fully-generic chain of
// `DimMin` / `Reallocator` allocator bounds required to call
// `col_piv_qr().unpack()` on `OMatrix<N, R, C>` is large enough to be a net
// readability loss without any practical benefit for Levenberg-Marquardt
// callers, who supply a runtime-sized Jacobian anyway. If a use case for
// statically-sized pivoted QR appears later, the trait surface already
// supports it via additional impls.
macro_rules! make_pivoted_qr {
    ($t:ty) => {
        impl ArgminPivotedQR for DMatrix<$t> {
            type Q = DMatrix<$t>;
            type R = DMatrix<$t>;
            type Permutation = PermutationSequence<Dyn>;
            type ColumnNorms = DVector<$t>;

            #[inline]
            fn qr_pivot(
                self,
            ) -> Result<PivotedQR<Self::Q, Self::R, Self::Permutation, Self::ColumnNorms>, Error>
            {
                let column_norms = DVector::<$t>::from_iterator(
                    self.ncols(),
                    (0..self.ncols()).map(|j| self.column(j).norm()),
                );
                let qr = self.col_piv_qr();
                let (q, r, permutation) = qr.unpack();
                Ok(PivotedQR {
                    q,
                    r,
                    permutation,
                    column_norms,
                })
            }
        }
    };
}

make_pivoted_qr!(f32);
make_pivoted_qr!(f64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ArgminPermute;
    use approx::assert_relative_eq;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_qr_pivot_reconstruct_ $t>]() {
                    let j = DMatrix::<$t>::from_row_slice(
                        4, 3,
                        &[
                            1.0, 0.1, 5.0,
                            2.0, 0.2, 6.0,
                            3.0, 0.3, 7.0,
                            4.0, 0.4, 8.0,
                        ],
                    );
                    let pqr = j.clone().qr_pivot().unwrap();

                    // J = Q · R · Pᵀ. Build Q*R, then apply Pᵀ on the right
                    // (i.e. column-permute Q*R back to the original col order
                    // — `inv_permute_columns` is exactly that).
                    let mut qr = &pqr.q * &pqr.r;
                    pqr.permutation.inv_permute_columns(&mut qr);
                    for i in 0..4 {
                        for k in 0..3 {
                            assert_relative_eq!(
                                qr[(i, k)],
                                j[(i, k)],
                                epsilon = (1e-5) as $t
                            );
                        }
                    }

                    // R is upper triangular.
                    for i in 1..3 {
                        for k in 0..i {
                            assert_relative_eq!(pqr.r[(i, k)], 0 as $t, epsilon = (1e-12) as $t);
                        }
                    }

                    // Q has orthonormal columns: Qᵀ Q = I
                    let qtq = pqr.q.transpose() * &pqr.q;
                    for i in 0..3 {
                        for k in 0..3 {
                            let expected = if i == k { 1 as $t } else { 0 as $t };
                            assert_relative_eq!(qtq[(i, k)], expected, epsilon = (1e-5) as $t);
                        }
                    }

                    // column_norms holds the L2 norms of the original columns.
                    for k in 0..3 {
                        let expected: $t = j.column(k).norm();
                        assert_relative_eq!(pqr.column_norms[k], expected, epsilon = (1e-5) as $t);
                    }
                }
            }
        };
    }

    make_test!(f32);
    make_test!(f64);
}
