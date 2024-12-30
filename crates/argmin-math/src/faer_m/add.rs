use crate::ArgminAdd;
use faer::{
    mat::{AsMatMut, AsMatRef},
    reborrow::{IntoConst, Reborrow, ReborrowMut},
    unzipped, zipped, zipped_rw, ComplexField, Conjugate, Entity, Mat, MatMut, MatRef,
    SimpleEntity,
};
use std::ops::{Add, AddAssign};

/// MatRef + Scalar -> Mat
impl<'a, E> ArgminAdd<E, Mat<E>> for MatRef<'a, E>
where
    E: Entity + Add<E, Output = E>,
{
    #[inline]
    fn add(&self, other: &E) -> Mat<E> {
        zipped_rw!(self).map(|unzipped!(this)| this.read() + *other)
    }
}

/// Scaler + MatRef-> Mat
impl<'a, E> ArgminAdd<MatRef<'a, E>, Mat<E>> for E
where
    E: Entity + Add<E, Output = E>,
{
    #[inline]
    fn add(&self, other: &MatRef<'a, E>) -> Mat<E> {
        // commutative with MatRef + Scalar so we can fall back on that case
        <_ as ArgminAdd<_, _>>::add(other, self)
    }
}

//@todo(geo) also add scalar + Matrix and matrix + Scalar (and reference variants?)

/// Mat + Scalar -> Mat
impl<E> ArgminAdd<E, Mat<E>> for Mat<E>
where
    E: Entity + Add<E, Output = E>,
{
    #[inline]
    fn add(&self, other: &E) -> Mat<E> {
        //@note(geo-ant) because we are taking self by reference we
        // cannot mutate the matrix in place, so we can just as well
        // reuse the reference code
        <_ as ArgminAdd<_, _>>::add(&self.as_mat_ref(), other)
    }
}

/// Scalar + Mat -> Mat
impl<E> ArgminAdd<Mat<E>, Mat<E>> for E
where
    E: Entity + Add<E, Output = E>,
{
    #[inline]
    fn add(&self, other: &Mat<E>) -> Mat<E> {
        // commutative with Mat + Scalar so we can fall back on that case
        <_ as ArgminAdd<_, _>>::add(other, self)
    }
}

/// MatRef + MatRef -> Mat
impl<'a, 'b, E: SimpleEntity + ComplexField> ArgminAdd<MatRef<'a, E>, Mat<E>> for MatRef<'b, E> {
    #[inline]
    fn add(&self, other: &MatRef<'a, E>) -> Mat<E> {
        <_ as Add>::add(self, other)
    }
}

/// MatRef + Mat -> Mat
impl<'a, 'b, E: SimpleEntity + ComplexField> ArgminAdd<Mat<E>, Mat<E>> for MatRef<'b, E> {
    #[inline]
    fn add(&self, other: &Mat<E>) -> Mat<E> {
        self + other
    }
}

/// Mat + Mat -> Mat
impl<'a, 'b, E: SimpleEntity + ComplexField> ArgminAdd<Mat<E>, Mat<E>> for Mat<E> {
    #[inline]
    fn add(&self, other: &Mat<E>) -> Mat<E> {
        self + other
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_scalar() {
        fn add_scalar(scalar: f64, mat: &Mat<f64>) -> Mat<f64> {
            <MatRef<f64> as ArgminAdd<f64, Mat<f64>>>::add(&mat.as_mat_ref(), &scalar)
        }
        let mat = Mat::<f64>::zeros(10, 11);
        let mut expected = Mat::<f64>::zeros(10, 11);
        let mat = add_scalar(1., &mat);
        expected.fill(1.);
        assert_eq!(mat, expected);
    }

    #[test]
    fn more_tests() {
        todo!()
    }
}
