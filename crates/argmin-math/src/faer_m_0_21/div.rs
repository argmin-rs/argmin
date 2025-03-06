use crate::ArgminDiv;
use faer::{
    mat::AsMatRef,
    reborrow::{IntoConst, Reborrow, ReborrowMut},
    unzip, zip, Mat, MatMut, MatRef,
};
use faer_traits::{ComplexField, DivByRef};
use std::ops::{Div, DivAssign};

/// MatRef / Scalar -> MatRef
impl<E> ArgminDiv<E, Mat<E>> for MatRef<'_, E>
where
    E: ComplexField + DivByRef<Output = E>,
{
    #[inline]
    fn div(&self, other: &E) -> Mat<E> {
        zip!(self).map(|unzip!(this)| this.div_by_ref(other))
    }
}

/// Mat / Scalar -> Mat
impl<E> ArgminDiv<E, Mat<E>> for Mat<E>
where
    E: ComplexField + DivByRef<Output = E>,
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
    E: ComplexField + DivByRef<Output = E>,
{
    #[inline]
    fn div(&self, other: &MatRef<'a, E>) -> Mat<E> {
        // does not commute with the expressions above, which is why
        // we need our own implementations
        zip!(other).map(|unzip!(other_elem)| self.div_by_ref(other_elem))
    }
}

/// Scalar / Mat -> Mat
impl<E> ArgminDiv<Mat<E>, Mat<E>> for E
where
    E: ComplexField + DivByRef<Output = E>,
{
    #[inline]
    fn div(&self, other: &Mat<E>) -> Mat<E> {
        //@note(geo-ant) because we are taking self by reference we
        // cannot mutate the matrix in place, so we can just as well
        // reuse the reference code
        <_ as ArgminDiv<_, _>>::div(self, &other.as_mat_ref())
    }
}

/// MatRef / MatRef -> Mat (pointwise division)
impl<'a, E: ComplexField + DivByRef<Output = E>> ArgminDiv<MatRef<'a, E>, Mat<E>>
    for MatRef<'_, E>
{
    #[inline]
    fn div(&self, other: &MatRef<'a, E>) -> Mat<E> {
        zip!(self, other).map(|unzip!(this, other)| this.div_by_ref(other))
    }
}

/// Mat / MatRef -> Mat (pointwise division)
impl<'a, E: ComplexField + DivByRef<Output = E>> ArgminDiv<MatRef<'a, E>, Mat<E>> for Mat<E> {
    #[inline]
    fn div(&self, other: &MatRef<'a, E>) -> Mat<E> {
        <_ as ArgminDiv<_, _>>::div(&self.as_mat_ref(), other)
    }
}

/// MatRef / Mat-> Mat (pointwise division)
impl<E: ComplexField + DivByRef<Output = E>> ArgminDiv<Mat<E>, Mat<E>> for MatRef<'_, E> {
    #[inline]
    fn div(&self, other: &Mat<E>) -> Mat<E> {
        <_ as ArgminDiv<_, _>>::div(self, &other.as_mat_ref())
    }
}

/// Mat / Mat-> Mat (pointwise division)
impl<E: ComplexField + DivByRef<Output = E>> ArgminDiv<Mat<E>, Mat<E>> for Mat<E> {
    #[inline]
    fn div(&self, other: &Mat<E>) -> Mat<E> {
        <_ as ArgminDiv<_, _>>::div(&self.as_mat_ref(), &other.as_mat_ref())
    }
}
