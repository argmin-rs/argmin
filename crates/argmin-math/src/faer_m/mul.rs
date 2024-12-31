use crate::ArgminMul;
use faer::{
    mat::{AsMatMut, AsMatRef},
    reborrow::{IntoConst, Reborrow, ReborrowMut},
    unzipped, zipped, zipped_rw, ComplexField, Conjugate, Entity, Mat, MatMut, MatRef,
    SimpleEntity,
};
use std::ops::Mul;

/// MatRef * Scalar -> Mat
impl<E> ArgminMul<E, Mat<E>> for MatRef<'_, E>
where
    E: Entity + Mul<E, Output = E>,
{
    #[inline]
    fn mul(&self, other: &E) -> Mat<E> {
        zipped_rw!(self).map(|unzipped!(this)| this.read() * *other)
    }
}

/// Scalar * MatRef-> Mat
impl<'a, E> ArgminMul<MatRef<'a, E>, Mat<E>> for E
where
    E: Entity + Mul<E, Output = E>,
{
    #[inline]
    fn mul(&self, other: &MatRef<'a, E>) -> Mat<E> {
        // commutative with MatRef + Scalar so we can fall back on that case
        <_ as ArgminMul<_, _>>::mul(other, self)
    }
}

/// Mat * Scalar -> Mat
impl<E> ArgminMul<E, Mat<E>> for Mat<E>
where
    E: Entity + Mul<E, Output = E>,
{
    #[inline]
    fn mul(&self, other: &E) -> Mat<E> {
        //@note(geo-ant) because we are taking self by reference we
        // cannot mutate the matrix in place, so we can just as well
        // reuse the reference code
        <_ as ArgminMul<_, _>>::mul(&self.as_mat_ref(), other)
    }
}

/// Scalar + Mat -> Mat
impl<E> ArgminMul<Mat<E>, Mat<E>> for E
where
    E: Entity + Mul<E, Output = E>,
{
    #[inline]
    fn mul(&self, other: &Mat<E>) -> Mat<E> {
        // commutative with Mat * Scalar so we can fall back on that case
        <_ as ArgminMul<_, _>>::mul(other, self)
    }
}

/// MatRef * MatRef -> Mat
impl<'a, E: SimpleEntity + ComplexField> ArgminMul<MatRef<'a, E>, Mat<E>> for MatRef<'_, E> {
    #[inline]
    fn mul(&self, other: &MatRef<'a, E>) -> Mat<E> {
        <_ as Mul>::mul(self, other)
    }
}

/// MatRef * Mat -> Mat
impl<'a, E: SimpleEntity + ComplexField> ArgminMul<Mat<E>, Mat<E>> for MatRef<'_, E> {
    #[inline]
    fn mul(&self, other: &Mat<E>) -> Mat<E> {
        self * other
    }
}

/// Mat * Mat -> Mat
impl<'a, 'b, E: SimpleEntity + ComplexField> ArgminMul<Mat<E>, Mat<E>> for Mat<E> {
    #[inline]
    fn mul(&self, other: &Mat<E>) -> Mat<E> {
        self * other
    }
}

#[cfg(test)]
mod test {

    #[test]
    fn more_tests() {
        todo!()
    }
}
