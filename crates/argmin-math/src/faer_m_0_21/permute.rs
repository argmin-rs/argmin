// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminPermute;
use faer::{perm::Perm, Mat};
use faer_traits::ComplexField;

impl<E: ComplexField> ArgminPermute<Mat<E>> for Perm<usize> {
    /// `result[j] = v[inverse[j]]`, i.e. `P · v` where `P_{forward[j], j} = 1`.
    #[inline]
    fn permute(&self, v: &Mat<E>) -> Mat<E> {
        let n = v.nrows();
        let ncols = v.ncols();
        let perm_ref = self.as_ref();
        assert_eq!(n, perm_ref.len());
        let (_, inverse) = perm_ref.arrays();
        Mat::<E>::from_fn(n, ncols, |i, j| v[(inverse[i], j)].clone())
    }

    /// `result[j] = v[forward[j]]`, i.e. `Pᵀ · v`.
    #[inline]
    fn unpermute(&self, v: &Mat<E>) -> Mat<E> {
        let n = v.nrows();
        let ncols = v.ncols();
        let perm_ref = self.as_ref();
        assert_eq!(n, perm_ref.len());
        let (forward, _) = perm_ref.arrays();
        Mat::<E>::from_fn(n, ncols, |i, j| v[(forward[i], j)].clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ArgminPivotedQR;
    use approx::assert_relative_eq;
    use faer::mat;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_permute_roundtrip_ $t>]() {
                    let j = mat![
                        [1.0 as $t, 0.1, 5.0],
                        [2.0 as $t, 0.2, 6.0],
                        [3.0 as $t, 0.3, 7.0],
                        [4.0 as $t, 0.4, 8.0]
                    ];
                    let pqr = j.qr_pivot().unwrap();
                    let v = mat![[10.0 as $t], [20.0], [30.0]];
                    let pv = pqr.permutation.permute(&v);
                    let v_back = pqr.permutation.unpermute(&pv);
                    for i in 0..3 {
                        assert_relative_eq!(v[(i, 0)], v_back[(i, 0)], epsilon = 1e-5 as $t);
                    }
                    let ptv = pqr.permutation.unpermute(&v);
                    let v_back2 = pqr.permutation.permute(&ptv);
                    for i in 0..3 {
                        assert_relative_eq!(v[(i, 0)], v_back2[(i, 0)], epsilon = 1e-5 as $t);
                    }
                }
            }
        };
    }

    make_test!(f32);
    make_test!(f64);
}
