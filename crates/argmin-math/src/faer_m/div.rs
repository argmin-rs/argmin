use crate::ArgminDiv;
use faer::{
    mat::AsMatRef,
    reborrow::{IntoConst, Reborrow, ReborrowMut},
    unzipped, zipped_rw, Conjugate, Entity, Mat, MatMut, MatRef, SimpleEntity,
};
use std::ops::{Div, DivAssign};

/// MatRef / Scalar -> MatRef
impl<E> ArgminDiv<E, Mat<E>> for MatRef<'_, E>
where
    E: Entity + Div<E, Output = E>,
{
    #[inline]
    fn div(&self, other: &E) -> Mat<E> {
        zipped_rw!(self).map(|unzipped!(this)| this.read() / *other)
    }
}

/// Mat / Scalar -> Mat
impl<E> ArgminDiv<E, Mat<E>> for Mat<E>
where
    E: Entity + Div<E, Output = E>,
{
    #[inline]
    fn div(&self, other: &E) -> Mat<E> {
        //@note(geo-ant) because we are taking self by reference we
        // cannot mutate the matrix in place, so we can just as well
        // reuse the reference code
        <_ as ArgminDiv<_, _>>::div(&self.as_mat_ref(), other)
    }
}

/// Scalar / MatRef -> Mat
impl<'a, E> ArgminDiv<MatRef<'a, E>, Mat<E>> for E
where
    E: Entity + Div<E, Output = E>,
{
    #[inline]
    fn div(&self, other: &MatRef<'a, E>) -> Mat<E> {
        // does not commute with the expressions above, which is why
        // we need our own implementations
        zipped_rw!(other).map(|unzipped!(other_elem)| *self / other_elem.read())
    }
}

/// Scalar / Mat -> Mat
impl<E> ArgminDiv<Mat<E>, Mat<E>> for E
where
    E: Entity + Div<E, Output = E>,
{
    #[inline]
    fn div(&self, other: &Mat<E>) -> Mat<E> {
        //@note(geo-ant) because we are taking self by reference we
        // cannot mutate the matrix in place, so we can just as well
        // reuse the reference code
        <_ as ArgminDiv<_, _>>::div(self, &other.as_mat_ref())
    }
}

/// MatRef / MatRef -> Mat
impl<'a, E: Entity + Div<E, Output = E>> ArgminDiv<MatRef<'a, E>, Mat<E>> for MatRef<'_, E> {
    #[inline]
    fn div(&self, other: &MatRef<'a, E>) -> Mat<E> {
        zipped_rw!(self, other).map(|unzipped!(this, other)| this.read() / other.read())
    }
}

/// Mat / MatRef -> Mat
impl<'a, E: Entity + Div<E, Output = E>> ArgminDiv<MatRef<'a, E>, Mat<E>> for Mat<E> {
    #[inline]
    fn div(&self, other: &MatRef<'a, E>) -> Mat<E> {
        <_ as ArgminDiv<_, _>>::div(&self.as_mat_ref(), other)
    }
}

/// MatRef / Mat-> Mat
impl<E: Entity + Div<E, Output = E>> ArgminDiv<Mat<E>, Mat<E>> for MatRef<'_, E> {
    #[inline]
    fn div(&self, other: &Mat<E>) -> Mat<E> {
        <_ as ArgminDiv<_, _>>::div(self, &other.as_mat_ref())
    }
}

/// Mat / Mat-> Mat
impl<E: Entity + Div<E, Output = E>> ArgminDiv<Mat<E>, Mat<E>> for Mat<E> {
    #[inline]
    fn div(&self, other: &Mat<E>) -> Mat<E> {
        <_ as ArgminDiv<_, _>>::div(&self.as_mat_ref(), &other.as_mat_ref())
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn more_tests() {
        todo!()
    }
}
