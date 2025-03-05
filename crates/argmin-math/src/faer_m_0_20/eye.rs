use crate::ArgminEye;
use faer::{ComplexField, Entity, Mat, Shape};

impl<E: Entity + ComplexField, R: Shape, C: Shape> ArgminEye for Mat<E, R, C>
where
    R: TryFrom<usize>,
    C: TryFrom<usize>,
{
    fn eye(n: usize) -> Self {
        let (nr, nc) = match (R::try_from(n), C::try_from(n)) {
            (Ok(nr), Ok(nc)) => (nr, nc),
            _ => panic!("invalid matrix size for index type"),
        };

        Mat::identity(nr, nc)
    }

    fn eye_like(&self) -> Self {
        let nr = self.nrows();
        let nc = self.ncols();
        //@note(geo-ant) in the nalgebra implementation we also enforce
        // that the matrix is square, which we don't enforce here
        Mat::identity(nr, nc)
    }
}
