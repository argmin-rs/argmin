use crate::ArgminL2Norm;
use faer::{ComplexField, Entity, Mat, MatRef, SimpleEntity};

impl<E: Entity + ComplexField> ArgminL2Norm<E::Real> for MatRef<'_, E> {
    fn l2_norm(&self) -> E::Real {
        self.norm_l2()
    }
}

impl<E: Entity + ComplexField> ArgminL2Norm<E::Real> for Mat<E> {
    fn l2_norm(&self) -> E::Real {
        self.norm_l2()
    }
}
