use crate::ArgminInv;
use faer::{
    dyn_stack::{GlobalPodBuffer, PodStack},
    linalg::lu::partial_pivoting::compute::{lu_in_place, lu_in_place_req},
    mat::{AsMatMut, AsMatRef},
    prelude::SolverCore,
    reborrow::ReborrowMut,
    Entity, Mat, MatRef, RealField, SimpleEntity,
};
use std::fmt;

#[derive(Debug, thiserror::Error, PartialEq)]
struct InverseError;

impl fmt::Display for InverseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Non-invertible matrix")
    }
}

/// calculate the inverse via LU decomposition with full pivoting
impl<E: SimpleEntity + RealField + PartialOrd> ArgminInv<Mat<E>> for MatRef<'_, E> {
    #[inline]
    fn inv(&self) -> Result<Mat<E>, anyhow::Error> {
        //@note(geo-ant) this panic is consistent with the
        // behavior when using nalgebra
        assert_eq!(
            self.nrows(),
            self.ncols(),
            "cannot invert non-square matrix"
        );
        //@note(geo-ant) to check whether the matrix is
        // invertible, we perform a rank-revealing decomposition
        // and check the diagonal elements of the appropriate matrix
        let lu_decomp = self.full_piv_lu(); //@note(geo-ant) rank revealing
        let umat = lu_decomp.compute_u();
        let is_singular = umat.diagonal().column_vector().iter().any(|elem: &E| {
            !elem.faer_is_finite() || (elem.faer_abs() <= E::faer_zero_threshold())
        });
        if !is_singular {
            Ok(lu_decomp.inverse())
        } else {
            Err(InverseError {}.into())
        }
    }
}

/// calculate the inverse via LU decomposition with full pivoting
impl<E: SimpleEntity + RealField> ArgminInv<Mat<E>> for Mat<E> {
    #[inline]
    fn inv(&self) -> Result<Mat<E>, anyhow::Error> {
        <_ as ArgminInv<_>>::inv(&self.as_mat_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::faer_tests::test_helper::*;
    use approx::assert_relative_eq;
    use faer::mat::AsMatRef;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_inv_ $t>]() {
                    let a = matrix2_new(
                        2 as $t, 5 as $t,
                        1 as $t, 3 as $t,
                    );
                    let target = matrix2_new(
                        3 as $t, -5 as $t,
                        -1 as $t, 2 as $t,
                    );
                    let res = <_ as ArgminInv<_>>::inv(&a).unwrap();
                    let res1 = <_ as ArgminInv<_>>::inv(&a.as_mat_ref()).unwrap();
                    assert_eq!(res,res1);
                    assert_eq!(res.nrows(),2);
                    assert_eq!(res.ncols(),2);
                    for i in 0..2 {
                        for j in 0..2 {
                            //@note(geo-ant) the 20 epsilon are a bit arbitrary,
                            // but it's to avoid spurious errors due to numerical effects
                            // while keeping good accuracy.
                            assert_relative_eq!(res[(i, j)], target[(i, j)], epsilon = 20.*$t::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_inv_error $t>]() {
                    let a = matrix2_new(
                        2 as $t, 5 as $t,
                        4 as $t, 10 as $t,
                    );
                    let err = <_ as ArgminInv<_>>::inv(&a).unwrap_err().downcast::<InverseError>().unwrap();
                    assert_eq!(err, InverseError {});
                }
            }
        };
    }

    make_test!(f32);
    make_test!(f64);
}
