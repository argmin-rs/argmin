use crate::ArgminTranspose;
use faer::{Conjugate, Entity, Mat, MatRef, Shape};

impl<'a, E, R, C> ArgminTranspose<MatRef<'a, E, C, R>> for MatRef<'a, E, R, C>
where
    E: Entity,
    R: Shape,
    C: Shape,
{
    #[inline]
    fn t(self) -> MatRef<'a, E, C, R> {
        self.transpose()
    }
}

impl<E> ArgminTranspose<Mat<E>> for Mat<E>
where
    E: Entity + Conjugate<Canonical = E>,
{
    #[inline]
    fn t(self) -> Mat<E> {
        self.transpose().to_owned()
    }
}

impl<'a, E> ArgminTranspose<MatRef<'a, E>> for &'a Mat<E>
where
    E: Entity,
{
    #[inline]
    fn t(self) -> MatRef<'a, E> {
        self.transpose()
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn add_tests() {
        todo!()
    }
}
