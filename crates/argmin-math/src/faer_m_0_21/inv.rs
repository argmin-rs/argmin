use crate::ArgminInv;
use faer::{
    linalg::solvers::DenseSolveCore,
    mat::{AsMatMut, AsMatRef},
    reborrow::ReborrowMut,
    Mat, MatRef,
};
use faer_traits::{math_utils, ComplexField, RealField};
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
impl<E: RealField + PartialOrd> ArgminInv<Mat<E>> for MatRef<'_, E> {
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
        let umat = lu_decomp.U();
        let is_singular = umat.diagonal().column_vector().iter().any(|elem: &E| {
            !math_utils::is_finite(elem)
                || (math_utils::abs(elem) <= math_utils::min_positive::<E>())
        });
        if !is_singular {
            Ok(lu_decomp.inverse())
        } else {
            Err(InverseError {}.into())
        }
    }
}

/// calculate the inverse via LU decomposition with full pivoting
impl<E: RealField> ArgminInv<Mat<E>> for Mat<E> {
    #[inline]
    fn inv(&self) -> Result<Mat<E>, anyhow::Error> {
        <_ as ArgminInv<_>>::inv(&self.as_mat_ref())
    }
}
