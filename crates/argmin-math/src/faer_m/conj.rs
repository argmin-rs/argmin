use crate::ArgminConj;
use faer::{mat::AsMatRef, reborrow::ReborrowMut, Conjugate, Entity, Mat, MatMut, MatRef};

use super::RealEntity;

impl<'a, E: Entity + Conjugate<Conj = E>> ArgminConj for MatRef<'a, E> {
    fn conj(&self) -> Self {
        self.conjugate()
    }
}

impl<E: RealEntity> ArgminConj for Mat<E> {
    #[inline]
    fn conj(&self) -> Self {
        self.as_mat_ref().conj().to_owned()
    }
}

#[cfg(test)]
mod test {
    use crate::ArgminConj;
    use faer::{
        mat::{AsMatMut, AsMatRef},
        Mat,
    };

    #[test]
    fn more_tests() {
        todo!()
    }

    #[test]
    fn test_conj() {
        let mat = Mat::<f64>::zeros(3, 2);
        let expected = mat.conjugate();
        assert_eq!(expected, <_ as ArgminConj>::conj(&mat.as_mat_ref()))
    }
}
