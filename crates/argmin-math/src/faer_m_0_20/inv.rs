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
/// an error during matrix inversion
pub struct InverseError;

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
