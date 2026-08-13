use crate::ArgminAdd;
use faer::{
    mat::{AsMatMut, AsMatRef},
    unzip, zip, Col, ColRef, Mat, MatMut, MatRef,
};
use faer_traits::{AddByRef, ComplexField};

/// ColRef + Scalar -> Col
impl<E> ArgminAdd<E, Col<E>> for ColRef<'_, E>
where
    E: ComplexField,
{
    #[inline]
    fn add(&self, other: &E) -> Col<E> {
        zip!(self).map(|unzip!(this)| this.add_by_ref(other))
    }
}

/// Scalar + ColRef-> Col
impl<'a, E> ArgminAdd<ColRef<'a, E>, Col<E>> for E
where
    E: ComplexField,
{
    #[inline]
    fn add(&self, other: &ColRef<'a, E>) -> Col<E> {
        // commutative with MatRef + Scalar so we can fall back on that case
        <_ as ArgminAdd<_, _>>::add(other, self)
    }
}

/// Col + Scalar -> Col
impl<E> ArgminAdd<E, Col<E>> for Col<E>
where
    E: ComplexField,
{
    #[inline]
    fn add(&self, other: &E) -> Col<E> {
        //@note(geo-ant) because we are taking self by reference we
        // cannot mutate the matrix in place, so we can just as well
        // reuse the reference code
        <_ as ArgminAdd<_, _>>::add(&self.as_ref(), other)
    }
}

/// Scalar + Col -> Col
impl<E> ArgminAdd<Col<E>, Col<E>> for E
where
    E: ComplexField,
{
    #[inline]
    fn add(&self, other: &Col<E>) -> Col<E> {
        // commutative with Col + Scalar so we can fall back on that case
        <_ as ArgminAdd<_, _>>::add(other, self)
    }
}

/// ColRef + ColRef -> Col
impl<'a, E> ArgminAdd<ColRef<'a, E>, Col<E>> for ColRef<'_, E>
where
    E: ComplexField,
{
    #[inline]
    fn add(&self, other: &ColRef<'a, E>) -> Col<E> {
        self + other
    }
}

/// ColRef + Col -> Col
impl<E: ComplexField> ArgminAdd<Col<E>, Col<E>> for ColRef<'_, E> {
    #[inline]
    fn add(&self, other: &Col<E>) -> Col<E> {
        self + other
    }
}

/// Col + ColRef -> Col
impl<E: ComplexField> ArgminAdd<ColRef<'_, E>, Col<E>> for Col<E> {
    #[inline]
    fn add(&self, other: &ColRef<'_, E>) -> Col<E> {
        self + other
    }
}

/// Col + Col -> Col
impl<E: ComplexField> ArgminAdd<Col<E>, Col<E>> for Col<E> {
    #[inline]
    fn add(&self, other: &Col<E>) -> Col<E> {
        self + other
    }
}
