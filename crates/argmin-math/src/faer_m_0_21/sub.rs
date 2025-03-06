use crate::ArgminSub;
use faer::{
    mat::AsMatRef,
    reborrow::{IntoConst, Reborrow, ReborrowMut},
    unzip, zip, Mat, MatMut, MatRef,
};
use faer_traits::{ComplexField, Conjugate};
use std::ops::{Sub, SubAssign};

/// MatRef / Scalar -> MatRef
impl<E> ArgminSub<E, Mat<E>> for MatRef<'_, E>
where
    E: ComplexField,
{
    #[inline]
    fn sub(&self, other: &E) -> Mat<E> {
        zip!(self).map(|unzip!(this)| this.sub_by_ref(other))
    }
}

/// Mat / Scalar -> Mat
impl<E> ArgminSub<E, Mat<E>> for Mat<E>
where
    E: ComplexField,
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
    E: ComplexField,
{
    #[inline]
    fn sub(&self, other: &MatRef<'a, E>) -> Mat<E> {
        // does not commute with the expressions above, which is why
        // we need our own implementations
        zip!(other).map(|unzip!(other_elem)| self.sub_by_ref(other_elem))
    }
}

/// Scalar / Mat -> Mat
impl<E> ArgminSub<Mat<E>, Mat<E>> for E
where
    E: ComplexField,
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
impl<'a, E: ComplexField> ArgminSub<MatRef<'a, E>, Mat<E>> for MatRef<'_, E> {
    #[inline]
    fn sub(&self, other: &MatRef<'a, E>) -> Mat<E> {
        zip!(self, other).map(|unzip!(this, other)| this.sub_by_ref(other))
    }
}

/// Mat / MatRef -> Mat
impl<'a, E: ComplexField> ArgminSub<MatRef<'a, E>, Mat<E>> for Mat<E> {
    #[inline]
    fn sub(&self, other: &MatRef<'a, E>) -> Mat<E> {
        <_ as ArgminSub<_, _>>::sub(&self.as_mat_ref(), other)
    }
}

/// MatRef / Mat-> Mat
impl<E: ComplexField> ArgminSub<Mat<E>, Mat<E>> for MatRef<'_, E> {
    #[inline]
    fn sub(&self, other: &Mat<E>) -> Mat<E> {
        <_ as ArgminSub<_, _>>::sub(self, &other.as_mat_ref())
    }
}

/// Mat / Mat-> Mat
impl<E: ComplexField> ArgminSub<Mat<E>, Mat<E>> for Mat<E> {
    #[inline]
    fn sub(&self, other: &Mat<E>) -> Mat<E> {
        <_ as ArgminSub<_, _>>::sub(&self.as_mat_ref(), &other.as_mat_ref())
    }
}
