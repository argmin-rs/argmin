use crate::ArgminL2Norm;
use faer::{Mat, MatRef};
use faer_traits::ComplexField;

impl<E: ComplexField> ArgminL2Norm<E::Real> for MatRef<'_, E> {
    fn l2_norm(&self) -> E::Real {
        self.norm_l2()
    }
}

impl<E: ComplexField> ArgminL2Norm<E::Real> for Mat<E> {
    fn l2_norm(&self) -> E::Real {
        self.norm_l2()
    }
}
