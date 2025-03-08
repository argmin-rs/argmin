use crate::ArgminTranspose;
use faer::{Mat, MatRef, Shape};
use faer_traits::Conjugate;

impl<'a, E, R, C> ArgminTranspose<MatRef<'a, E, C, R>> for MatRef<'a, E, R, C>
where
    R: Shape,
    C: Shape,
{
    #[inline]
    fn t(self) -> MatRef<'a, E, C, R> {
        self.transpose()
    }
}

impl<E> ArgminTranspose<Mat<E>> for MatRef<'_, E>
where
    E: Conjugate<Canonical = E>,
{
    #[inline]
    fn t(self) -> Mat<E> {
        self.transpose().to_owned()
    }
}

impl<E> ArgminTranspose<Mat<E>> for Mat<E>
where
    E: Conjugate<Canonical = E>,
{
    #[inline]
    fn t(self) -> Mat<E> {
        self.transpose().to_owned()
    }
}

impl<'a, E> ArgminTranspose<MatRef<'a, E>> for &'a Mat<E> {
    #[inline]
    fn t(self) -> MatRef<'a, E> {
        self.transpose()
    }
}
