use crate::ArgminSub;
use faer::{
    mat::AsMatRef,
    reborrow::{IntoConst, Reborrow, ReborrowMut},
    unzipped, zipped_rw, Conjugate, Entity, Mat, MatMut, MatRef, SimpleEntity,
};
use std::ops::{Sub, SubAssign};

/// MatRef / Scalar -> MatRef
impl<'a, E> ArgminSub<E, Mat<E>> for MatRef<'a, E>
where
    E: Entity + Sub<E, Output = E>,
{
    #[inline]
    fn sub(&self, other: &E) -> Mat<E> {
        zipped_rw!(self).map(|unzipped!(this)| this.read() - *other)
    }
}

/// Mat / Scalar -> Mat
impl<E> ArgminSub<E, Mat<E>> for Mat<E>
where
    E: Entity + Sub<E, Output = E>,
{
    #[inline]
    fn sub(&self, other: &E) -> Mat<E> {
        //@note(geo-ant) because we are taking self by reference we
        // cannot mutate the matrix in place, so we can just as well
        // reuse the reference code
        <_ as ArgminSub<_, _>>::sub(&self.as_mat_ref(), other)
    }
}

/// Scalar / MatRef -> Mat
impl<'a, E> ArgminSub<MatRef<'a, E>, Mat<E>> for E
where
    E: Entity + Sub<E, Output = E>,
{
    #[inline]
    fn sub(&self, other: &MatRef<'a, E>) -> Mat<E> {
        // does not commute with the expressions above, which is why
        // we need our own implementations
        zipped_rw!(other).map(|unzipped!(other_elem)| *self - other_elem.read())
    }
}

/// Scalar / Mat -> Mat
impl<E> ArgminSub<Mat<E>, Mat<E>> for E
where
    E: Entity + Sub<E, Output = E>,
{
    #[inline]
    fn sub(&self, other: &Mat<E>) -> Mat<E> {
        //@note(geo-ant) because we are taking self by reference we
        // cannot mutate the matrix in place, so we can just as well
        // reuse the reference code
        <_ as ArgminSub<_, _>>::sub(self, &other.as_mat_ref())
    }
}

/// MatRef / MatRef -> Mat
impl<'a, 'b, E: Entity + Sub<E, Output = E>> ArgminSub<MatRef<'a, E>, Mat<E>> for MatRef<'b, E> {
    #[inline]
    fn sub(&self, other: &MatRef<'a, E>) -> Mat<E> {
        zipped_rw!(self, other).map(|unzipped!(this, other)| this.read() - other.read())
    }
}

/// Mat / MatRef -> Mat
impl<'a, E: Entity + Sub<E, Output = E>> ArgminSub<MatRef<'a, E>, Mat<E>> for Mat<E> {
    #[inline]
    fn sub(&self, other: &MatRef<'a, E>) -> Mat<E> {
        <_ as ArgminSub<_, _>>::sub(&self.as_mat_ref(), other)
    }
}

/// MatRef / Mat-> Mat
impl<'a, E: Entity + Sub<E, Output = E>> ArgminSub<Mat<E>, Mat<E>> for MatRef<'a, E> {
    #[inline]
    fn sub(&self, other: &Mat<E>) -> Mat<E> {
        <_ as ArgminSub<_, _>>::sub(self, &other.as_mat_ref())
    }
}

/// Mat / Mat-> Mat
impl<E: Entity + Sub<E, Output = E>> ArgminSub<Mat<E>, Mat<E>> for Mat<E> {
    #[inline]
    fn sub(&self, other: &Mat<E>) -> Mat<E> {
        <_ as ArgminSub<_, _>>::sub(&self.as_mat_ref(), &other.as_mat_ref())
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn more_tests() {
        todo!()
    }
}
