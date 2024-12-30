use crate::ArgminDot;
use faer::{ComplexField, Mat, MatRef, SimpleEntity};
use std::ops::Mul;

//@note(geo): the order is important here.
// the way it is implemented with nalgebra suggests that this calculates
// self.transpose() * other
// which is in contrast to the documentation of the trait itself,
// where it says "dot product of T and Self"

/// MatRef . MatRef -> Mat
impl<'a, E: SimpleEntity + ComplexField> ArgminDot<MatRef<'a, E>, Mat<E>> for MatRef<'a, E> {
    #[inline]
    fn dot(&self, other: &MatRef<'a, E>) -> Mat<E> {
        self.transpose() * other
    }
}

/// MatRef . MatRef -> Mat
impl<'a, E: SimpleEntity + ComplexField> ArgminDot<Mat<E>, Mat<E>> for MatRef<'a, E> {
    #[inline]
    fn dot(&self, other: &Mat<E>) -> Mat<E> {
        self.transpose() * other
    }
}

/// Mat . MatRef -> Mat
impl<'a, E: SimpleEntity + ComplexField> ArgminDot<Mat<E>, Mat<E>> for Mat<E> {
    #[inline]
    fn dot(&self, other: &Mat<E>) -> Mat<E> {
        self.transpose() * other
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn more_tests() {
        todo!()
    }
}
