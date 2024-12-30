use crate::ArgminEye;
use faer::{ComplexField, Entity, Mat};

impl<E: Entity + ComplexField> ArgminEye for Mat<E> {
    fn eye(n: usize) -> Self {
        Mat::<_>::identity(n, n)
    }

    fn eye_like(&self) -> Self {
        let n = self.nrows();
        //@note(geo-ant) this constraint is enforced in the nalgebra implementation.
        // faer does not need it, but I felt it's better to keep the same runtime
        // invariants.
        assert_eq!(n, self.ncols(), "internal error: expected square matrix");
        Mat::<_>::identity(n, n)
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn more_tests() {
        todo!()
    }
}
