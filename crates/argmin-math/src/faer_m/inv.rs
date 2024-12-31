use crate::ArgminInv;
use faer::{
    dyn_stack::{GlobalPodBuffer, PodStack},
    linalg::lu::partial_pivoting::compute::{lu_in_place, lu_in_place_req},
    mat::{AsMatMut, AsMatRef},
    prelude::SolverCore,
    reborrow::ReborrowMut,
    ComplexField, Entity, Mat, MatRef, SimpleEntity,
};
use std::fmt;

#[derive(Debug, thiserror::Error, PartialEq)]
struct InverseError;

impl fmt::Display for InverseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Non-invertible matrix")
    }
}

/// calculate the inverse by using LU decomposition with partial pivoting
impl<E: Entity + ComplexField> ArgminInv<Mat<E>> for MatRef<'_, E> {
    #[inline]
    fn inv(&self) -> Result<Mat<E>, anyhow::Error> {
        if self.nrows() != self.ncols() || self.nrows() == 0 {
            Err(InverseError.into())
        } else {
            Ok(self.partial_piv_lu().inverse())
        }
    }
}

/// calculate the inverse by using LU decomposition with partial pivoting
impl<'a, E: Entity + ComplexField> ArgminInv<Mat<E>> for Mat<E> {
    #[inline]
    fn inv(&self) -> Result<Mat<E>, anyhow::Error> {
        if self.nrows() != self.ncols() || self.nrows() == 0 {
            Err(InverseError.into())
        } else {
            Ok(self.partial_piv_lu().inverse())
        }
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn more_tests() {
        todo!()
    }
}
