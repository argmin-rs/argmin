use crate::ArgminDot;
use faer::{mat::AsMatRef, Mat, MatRef};
use faer_traits::ComplexField;
use std::ops::Mul;

//@note(geo): the order is important here.
// the way it is implemented with nalgebra suggests that this calculates
// self.conjugate() * other
// which is in contrast to the documentation of the trait itself,
// where it says "dot product of T and Self"

/// ArgminDot implementation for matrix multiplication: Matrix . Matrix -> Matrix
/// In these cases, ArgminDot means the matrix-matrix product
mod matrix_matrix_multiplication {
    use super::*;

    /// MatRef . MatRef -> Mat
    impl<'a, E: ComplexField> ArgminDot<MatRef<'a, E>, Mat<E>> for MatRef<'_, E> {
        #[inline]
        fn dot(&self, other: &MatRef<'a, E>) -> Mat<E> {
            self * other
        }
    }

    /// MatRef . Mat -> Mat
    impl<E: ComplexField> ArgminDot<Mat<E>, Mat<E>> for MatRef<'_, E> {
        #[inline]
        fn dot(&self, other: &Mat<E>) -> Mat<E> {
            <_ as ArgminDot<_, _>>::dot(self, &other.as_mat_ref())
        }
    }

    /// Mat . MatRef -> Mat
    impl<'a, E: ComplexField> ArgminDot<MatRef<'a, E>, Mat<E>> for Mat<E> {
        #[inline]
        fn dot(&self, other: &MatRef<'a, E>) -> Mat<E> {
            <_ as ArgminDot<_, _>>::dot(&self.as_mat_ref(), other)
        }
    }

    /// Mat . Mat -> Mat
    impl<E: ComplexField> ArgminDot<Mat<E>, Mat<E>> for Mat<E> {
        #[inline]
        fn dot(&self, other: &Mat<E>) -> Mat<E> {
            <_ as ArgminDot<_, _>>::dot(&self.as_mat_ref(), &other.as_mat_ref())
        }
    }
}

/// contains implementations for the scalar product of two column vectors of
/// the same length. This is v^H . u for two column vectors v,u.
//@note(geo-ant) the corresponding nalgebra implementations allow taking a scalar
// product of any two matrices of same shape ("as vectors"). I've opted to not
// reproduce this behavior here, since it's likely invoked in error.
mod scalar_product {
    use faer_traits::Conjugate;

    use super::*;
    /// MatRef . MatRef -> Mat
    impl<'a, E: ComplexField + Conjugate<Conj = E>> ArgminDot<MatRef<'a, E>, E> for MatRef<'_, E> {
        #[inline]
        fn dot(&self, other: &MatRef<'a, E>) -> E {
            //@note(geo): we allow the scalar dot product between two vectors
            // of same length (but possibly different shape).
            assert!(
                (self.nrows() == 1 || self.ncols() == 1)
                    && (other.nrows() == 1 || other.ncols() == 1),
                "arguments for dot product must be vectors"
            );
            let count = std::cmp::max(self.nrows(), self.ncols());
            let count_rhs = std::cmp::max(other.nrows(), other.ncols());
            assert_eq!(
                count, count_rhs,
                "vectors for dot product must have same number of elements"
            );

            let value: Mat<E> = <_ as ArgminDot<_, _>>::dot(
                &self.as_shape(count, 1).conjugate().transpose(),
                &other.as_shape(count, 1),
            );
            debug_assert_eq!(value.nrows(), 1);
            debug_assert_eq!(value.ncols(), 1);
            value[(0, 0)].clone()
        }
    }

    /// MatRef . Mat -> Mat
    impl<E: ComplexField + Conjugate<Conj = E>> ArgminDot<Mat<E>, E> for MatRef<'_, E> {
        #[inline]
        fn dot(&self, other: &Mat<E>) -> E {
            <_ as ArgminDot<_, _>>::dot(self, &other.as_mat_ref())
        }
    }

    /// Mat . MatRef -> Mat
    impl<'a, E: ComplexField + Conjugate<Conj = E>> ArgminDot<MatRef<'a, E>, E> for Mat<E> {
        #[inline]
        fn dot(&self, other: &MatRef<'a, E>) -> E {
            <_ as ArgminDot<_, _>>::dot(&self.as_mat_ref(), other)
        }
    }

    /// Mat . Mat -> Mat
    impl<E: ComplexField + Conjugate<Conj = E>> ArgminDot<Mat<E>, E> for Mat<E> {
        #[inline]
        fn dot(&self, other: &Mat<E>) -> E {
            <_ as ArgminDot<_, _>>::dot(&self.as_mat_ref(), &other.as_mat_ref())
        }
    }
}

//@note(geo) implemented for compatibility with the nalgebra implementations,
// but this should probably not have to exist, since the functionality is
// already covered with ArgminMul
// Scalar . Matrix -> Matrix
// and Matrix . Scalar -> Matrix
mod multiply_matrix_with_scalar {
    use super::*;
    use crate::ArgminMul;
    use faer_traits::ComplexField;
    use std::ops::Mul;

    // MatRef . Scalar -> Mat
    impl<E: ComplexField> ArgminDot<E, Mat<E>> for MatRef<'_, E> {
        #[inline]
        fn dot(&self, other: &E) -> Mat<E> {
            <Self as ArgminMul<E, _>>::mul(self, other)
        }
    }

    // Mat . Scalar -> Mat
    impl<E: ComplexField> ArgminDot<E, Mat<E>> for Mat<E> {
        #[inline]
        fn dot(&self, other: &E) -> Mat<E> {
            <_ as ArgminDot<E, _>>::dot(&self.as_mat_ref(), other)
        }
    }

    // MatRef . Scalar -> Mat
    impl<'a, E: ComplexField> ArgminDot<MatRef<'a, E>, Mat<E>> for E {
        #[inline]
        fn dot(&self, other: &MatRef<'a, E>) -> Mat<E> {
            <E as ArgminMul<MatRef<'a, E>, _>>::mul(self, other)
        }
    }

    // Mat . Scalar -> Mat
    impl<E: ComplexField> ArgminDot<Mat<E>, Mat<E>> for E {
        #[inline]
        fn dot(&self, other: &Mat<E>) -> Mat<E> {
            <E as ArgminDot<_, _>>::dot(self, &other.as_mat_ref())
        }
    }
}
