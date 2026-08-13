use crate::ArgminDot;
use faer::prelude::*;
use faer_traits::ComplexField;

/// contains implementations for applying matrices to column vectors.
mod matrix_column_multiplication {
    use super::*;

    /// MatRef . ColRef -> Col
    impl<E: ComplexField> ArgminDot<ColRef<'_, E>, Col<E>> for MatRef<'_, E> {
        #[inline]
        fn dot(&self, other: &ColRef<'_, E>) -> Col<E> {
            self * other
        }
    }

    /// Mat . ColRef -> Col
    impl<E: ComplexField> ArgminDot<ColRef<'_, E>, Col<E>> for Mat<E> {
        #[inline]
        fn dot(&self, other: &ColRef<'_, E>) -> Col<E> {
            <_ as ArgminDot<_, _>>::dot(&self.as_ref(), other)
        }
    }

    /// MatRef . ColRef -> Col
    impl<E: ComplexField> ArgminDot<Col<E>, Col<E>> for MatRef<'_, E> {
        #[inline]
        fn dot(&self, other: &Col<E>) -> Col<E> {
            <_ as ArgminDot<_, _>>::dot(self, &other.as_ref())
        }
    }

    /// Mat . Col -> Col
    impl<E: ComplexField> ArgminDot<Col<E>, Col<E>> for Mat<E> {
        #[inline]
        fn dot(&self, other: &Col<E>) -> Col<E> {
            <_ as ArgminDot<_, _>>::dot(&self.as_ref(), &other.as_ref())
        }
    }
}

/// contains implementations for the scalar product of two column vectors of
/// the same length. This is v^H . u for two column vectors v,u.
mod scalar_product {
    use super::*;
    use faer_traits::Conjugate;

    /// ColRef . ColRef -> Scalar
    impl<E: ComplexField + Conjugate<Conj = E>> ArgminDot<ColRef<'_, E>, E> for ColRef<'_, E> {
        #[inline]
        fn dot(&self, other: &ColRef<'_, E>) -> E {
            assert_eq!(
                self.nrows(),
                other.nrows(),
                "vectors for dot product must have same number of elements"
            );
            self.conjugate().transpose() * other
        }
    }

    /// Col . ColRef -> Scalar
    impl<E: ComplexField + Conjugate<Conj = E>> ArgminDot<ColRef<'_, E>, E> for Col<E> {
        #[inline]
        fn dot(&self, other: &ColRef<'_, E>) -> E {
            <_ as ArgminDot<_, _>>::dot(&self.as_ref(), other)
        }
    }

    /// ColRef . Col -> Scalar
    impl<E: ComplexField + Conjugate<Conj = E>> ArgminDot<Col<E>, E> for ColRef<'_, E> {
        #[inline]
        fn dot(&self, other: &Col<E>) -> E {
            <_ as ArgminDot<_, _>>::dot(self, &other.as_ref())
        }
    }

    /// Col . Col -> Scalar
    impl<E: ComplexField + Conjugate<Conj = E>> ArgminDot<Col<E>, E> for Col<E> {
        #[inline]
        fn dot(&self, other: &Col<E>) -> E {
            <_ as ArgminDot<_, _>>::dot(&self.as_ref(), &other.as_ref())
        }
    }
}

mod outer_col_product {
    use super::*;

    /// ColRef . ColRef -> Mat
    impl<E: ComplexField> ArgminDot<ColRef<'_, E>, Mat<E>> for ColRef<'_, E> {
        #[inline]
        fn dot(&self, other: &ColRef<'_, E>) -> Mat<E> {
            Mat::from_fn(self.nrows(), other.nrows(), |i, j| &self[i] * &other[j])
        }
    }

    /// Col . ColRef -> Mat
    impl<E: ComplexField> ArgminDot<ColRef<'_, E>, Mat<E>> for Col<E> {
        #[inline]
        fn dot(&self, other: &ColRef<'_, E>) -> Mat<E> {
            <_ as ArgminDot<_, _>>::dot(&self.as_ref(), other)
        }
    }

    /// ColRef . Col -> Mat
    impl<E: ComplexField> ArgminDot<Col<E>, Mat<E>> for ColRef<'_, E> {
        #[inline]
        fn dot(&self, other: &Col<E>) -> Mat<E> {
            <_ as ArgminDot<_, _>>::dot(self, &other.as_ref())
        }
    }

    /// Col . Col -> Mat
    impl<E: ComplexField> ArgminDot<Col<E>, Mat<E>> for Col<E> {
        #[inline]
        fn dot(&self, other: &Col<E>) -> Mat<E> {
            <_ as ArgminDot<_, _>>::dot(&self.as_ref(), &other.as_ref())
        }
    }
}

//@note(clouds) implemented for compatibility with the nalgebra implementations
// see geo's comment in the faer_m_0_21 module (super)
mod multiply_col_with_scalar {
    use super::*;
    use crate::ArgminMul;
    use faer_traits::ComplexField;
    use std::ops::Mul;

    // ColRef . Scalar -> Col
    impl<E: ComplexField> ArgminDot<E, Col<E>> for ColRef<'_, E> {
        #[inline]
        fn dot(&self, other: &E) -> Col<E> {
            <Self as ArgminMul<E, _>>::mul(self, other)
        }
    }

    // Col . Scalar -> Col
    impl<E: ComplexField> ArgminDot<E, Col<E>> for Col<E> {
        #[inline]
        fn dot(&self, other: &E) -> Col<E> {
            <_ as ArgminDot<E, _>>::dot(&self.as_ref(), other)
        }
    }

    // Scalar . ColRef  -> Col
    impl<'a, E: ComplexField> ArgminDot<ColRef<'a, E>, Col<E>> for E {
        #[inline]
        fn dot(&self, other: &ColRef<'a, E>) -> Col<E> {
            <E as ArgminMul<ColRef<'a, E>, _>>::mul(self, other)
        }
    }

    // Scalar . Col -> Col
    impl<E: ComplexField> ArgminDot<Col<E>, Col<E>> for E {
        #[inline]
        fn dot(&self, other: &Col<E>) -> Col<E> {
            <E as ArgminDot<_, _>>::dot(self, &other.as_ref())
        }
    }
}
