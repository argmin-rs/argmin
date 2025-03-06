use crate::ArgminL1Norm;
use faer::mat::AsMatRef;
use faer::{Mat, MatRef};
use faer_traits::ComplexField;

impl<E: ComplexField> ArgminL1Norm<E::Real> for MatRef<'_, E> {
    fn l1_norm(&self) -> E::Real {
        self.norm_l1()
    }
}

impl<E: ComplexField> ArgminL1Norm<E::Real> for Mat<E> {
    fn l1_norm(&self) -> E::Real {
        self.as_mat_ref().norm_l1()
    }
}
