use crate::ArgminZeroLike;
use faer::{Mat, Shape};

impl<E, R, C> ArgminZeroLike for Mat<E, R, C>
where
    E: faer_traits::ComplexField,
    R: Shape,
    C: Shape,
{
    fn zero_like(&self) -> Self {
        Self::zeros(self.nrows(), self.ncols())
    }
}
