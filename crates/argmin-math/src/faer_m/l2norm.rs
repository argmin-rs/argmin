use crate::ArgminL2Norm;
use faer::{ComplexField, Entity, Mat, MatRef, SimpleEntity};

impl<'a, E: Entity + ComplexField> ArgminL2Norm<E::Real> for MatRef<'a, E> {
    fn l2_norm(&self) -> E::Real {
        self.norm_l2()
    }
}

impl<'a, E: Entity + ComplexField> ArgminL2Norm<E::Real> for Mat<E> {
    fn l2_norm(&self) -> E::Real {
        self.norm_l2()
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn more_tests() {
        todo!()
    }
}
