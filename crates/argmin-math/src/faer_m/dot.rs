use crate::ArgminDot;
use faer::{mat::AsMatRef, ComplexField, Mat, MatRef, SimpleEntity};
use std::ops::Mul;

//@note(geo): the order is important here.
// the way it is implemented with nalgebra suggests that this calculates
// self.transpose() * other
// which is in contrast to the documentation of the trait itself,
// where it says "dot product of T and Self"

/// MatRef . MatRef -> Mat
impl<'a, E: SimpleEntity + ComplexField> ArgminDot<MatRef<'a, E>, Mat<E>> for MatRef<'_, E> {
    #[inline]
    fn dot(&self, other: &MatRef<'a, E>) -> Mat<E> {
        //@note(geo-ant) maybe this would be faster using the matmul with conjugation
        self.conjugate() * other
    }
}

/// MatRef . Mat -> Mat
impl<E: SimpleEntity + ComplexField> ArgminDot<Mat<E>, Mat<E>> for MatRef<'_, E> {
    #[inline]
    fn dot(&self, other: &Mat<E>) -> Mat<E> {
        <_ as ArgminDot<_, _>>::dot(self, &other.as_mat_ref())
    }
}

/// Mat . MatRef -> Mat
impl<'a, E: SimpleEntity + ComplexField> ArgminDot<MatRef<'a, E>, Mat<E>> for Mat<E> {
    #[inline]
    fn dot(&self, other: &MatRef<'a, E>) -> Mat<E> {
        <_ as ArgminDot<_, _>>::dot(&self.as_mat_ref(), other)
    }
}

/// Mat . Mat -> Mat
impl<E: SimpleEntity + ComplexField> ArgminDot<Mat<E>, Mat<E>> for Mat<E> {
    #[inline]
    fn dot(&self, other: &Mat<E>) -> Mat<E> {
        <_ as ArgminDot<_, _>>::dot(&self.as_mat_ref(), &other.as_mat_ref())
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn more_tests() {
        todo!()
    }
}
