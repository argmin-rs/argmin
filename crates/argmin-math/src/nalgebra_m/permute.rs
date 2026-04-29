// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use super::Dyn;
use crate::ArgminPermute;
use nalgebra::{DVector, PermutationSequence};

// Note: nalgebra's `PermutationSequence::permute_rows` applies `Pᵀ` and
// `inv_permute_rows` applies `P` (verified empirically against the
// reconstruction `J = Q · R · Pᵀ` produced by `col_piv_qr`).
macro_rules! make_permute {
    ($t:ty) => {
        impl ArgminPermute<DVector<$t>> for PermutationSequence<Dyn> {
            #[inline]
            fn permute(&self, v: &DVector<$t>) -> DVector<$t> {
                let mut out = v.clone();
                self.inv_permute_rows(&mut out);
                out
            }

            #[inline]
            fn unpermute(&self, v: &DVector<$t>) -> DVector<$t> {
                let mut out = v.clone();
                self.permute_rows(&mut out);
                out
            }
        }
    };
}

make_permute!(f32);
make_permute!(f64);

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::DMatrix;
    use paste::item;

    use crate::ArgminPivotedQR;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_permute_roundtrip_ $t>]() {
                    let j = DMatrix::<$t>::from_row_slice(
                        4, 3,
                        &[
                            1.0, 0.1, 5.0,
                            2.0, 0.2, 6.0,
                            3.0, 0.3, 7.0,
                            4.0, 0.4, 8.0,
                        ],
                    );
                    let pqr = j.qr_pivot().unwrap();
                    let v = DVector::<$t>::from_vec(vec![10.0, 20.0, 30.0]);
                    let pv = pqr.permutation.permute(&v);
                    let v_back = pqr.permutation.unpermute(&pv);
                    for i in 0..3 {
                        assert_relative_eq!(v[i], v_back[i], epsilon = (1e-12) as $t);
                    }
                    let ptv = pqr.permutation.unpermute(&v);
                    let v_back2 = pqr.permutation.permute(&ptv);
                    for i in 0..3 {
                        assert_relative_eq!(v[i], v_back2[i], epsilon = (1e-12) as $t);
                    }
                }
            }
        };
    }

    make_test!(f32);
    make_test!(f64);
}
