use faer::{Entity, Mat, Shape};

use crate::ArgminZeroLike;

impl<E, R, C> ArgminZeroLike for Mat<E, R, C>
where
    E: Entity,
    R: Shape,
    C: Shape,
{
    fn zero_like(&self) -> Self {
        Self::zeros(self.nrows(), self.ncols())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn add_tests() {
        todo!()
    }
}
