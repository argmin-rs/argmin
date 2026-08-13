use crate::ArgminL2Norm;
use faer::{Col, ColRef};
use faer_traits::ComplexField;

impl<E: ComplexField> ArgminL2Norm<E::Real> for ColRef<'_, E> {
    fn l2_norm(&self) -> E::Real {
        self.norm_l2()
    }
}

impl<E: ComplexField> ArgminL2Norm<E::Real> for Col<E> {
    fn l2_norm(&self) -> E::Real {
        self.norm_l2()
    }
}
