use crate::ArgminSub;
use faer::{unzip, zip, Col, ColMut, ColRef};
use faer_traits::{ComplexField, Conjugate, SubByRef};

/// ColRef / Scalar -> Col
impl<E: ComplexField> ArgminSub<E, Col<E>> for ColRef<'_, E> {
    #[inline]
    fn sub(&self, other: &E) -> Col<E> {
        zip!(self).map(|unzip!(this)| this.sub_by_ref(other))
    }
}

/// Col / Scalar -> Col
impl<E: ComplexField> ArgminSub<E, Col<E>> for Col<E> {
    #[inline]
    fn sub(&self, other: &E) -> Col<E> {
        <_ as ArgminSub<_, _>>::sub(&self.as_ref(), other)
    }
}

/// Scalar / ColRef -> Col
impl<'a, E> ArgminSub<ColRef<'a, E>, Col<E>> for E
where
    E: ComplexField,
{
    #[inline]
    fn sub(&self, other: &ColRef<'a, E>) -> Col<E> {
        zip!(other).map(|unzip!(other)| self.sub_by_ref(other))
    }
}
/// Scalar / Col -> Col
impl<E> ArgminSub<Col<E>, Col<E>> for E
where
    E: ComplexField,
{
    #[inline]
    fn sub(&self, other: &Col<E>) -> Col<E> {
        <_ as ArgminSub<_, _>>::sub(self, &other.as_ref())
    }
}

/// ColRef / ColRef -> Col
impl<E: ComplexField> ArgminSub<ColRef<'_, E>, Col<E>> for ColRef<'_, E> {
    #[inline]
    fn sub(&self, other: &ColRef<'_, E>) -> Col<E> {
        zip!(self, other).map(|unzip!(this, other)| this.sub_by_ref(other))
    }
}

/// Col / ColRef -> Col
impl<E: ComplexField> ArgminSub<ColRef<'_, E>, Col<E>> for Col<E> {
    #[inline]
    fn sub(&self, other: &ColRef<'_, E>) -> Col<E> {
        <_ as ArgminSub<_, _>>::sub(&self.as_ref(), other)
    }
}

/// ColRef / Col -> Col
impl<E: ComplexField> ArgminSub<Col<E>, Col<E>> for ColRef<'_, E> {
    #[inline]
    fn sub(&self, other: &Col<E>) -> Col<E> {
        <_ as ArgminSub<_, _>>::sub(self, &other.as_ref())
    }
}

/// Col / Col -> Col
impl<E: ComplexField> ArgminSub<Col<E>, Col<E>> for Col<E> {
    #[inline]
    fn sub(&self, other: &Col<E>) -> Col<E> {
        <_ as ArgminSub<_, _>>::sub(&self.as_ref(), &other.as_ref())
    }
}
