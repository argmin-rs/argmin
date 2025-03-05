use crate::ArgminDot;
use faer::{mat::AsMatRef, ComplexField, Mat, MatRef, SimpleEntity};
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
    impl<'a, E: SimpleEntity + ComplexField> ArgminDot<MatRef<'a, E>, Mat<E>> for MatRef<'_, E> {
        #[inline]
        fn dot(&self, other: &MatRef<'a, E>) -> Mat<E> {
            self * other
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
}

/// contains implementations for the scalar product of two column vectors of
/// the same length. This is v^H . u for two column vectors v,u.
//@note(geo-ant) the corresponding nalgebra implementations allow taking a scalar
// product of any two matrices of same shape ("as vectors"). I've opted to not
// reproduce this behavior here, since it's likely invoked in error.
mod scalar_product {
    use faer::Conjugate;

    use super::*;
    /// MatRef . MatRef -> Mat
    impl<'a, E: SimpleEntity + ComplexField + Conjugate<Conj = E>> ArgminDot<MatRef<'a, E>, E>
        for MatRef<'_, E>
    {
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
            value[(0, 0)]
        }
    }

    /// MatRef . Mat -> Mat
    impl<E: SimpleEntity + ComplexField + Conjugate<Conj = E>> ArgminDot<Mat<E>, E> for MatRef<'_, E> {
        #[inline]
        fn dot(&self, other: &Mat<E>) -> E {
            <_ as ArgminDot<_, _>>::dot(self, &other.as_mat_ref())
        }
    }

    /// Mat . MatRef -> Mat
    impl<'a, E: SimpleEntity + ComplexField + Conjugate<Conj = E>> ArgminDot<MatRef<'a, E>, E>
        for Mat<E>
    {
        #[inline]
        fn dot(&self, other: &MatRef<'a, E>) -> E {
            <_ as ArgminDot<_, _>>::dot(&self.as_mat_ref(), other)
        }
    }

    /// Mat . Mat -> Mat
    impl<E: SimpleEntity + ComplexField + Conjugate<Conj = E>> ArgminDot<Mat<E>, E> for Mat<E> {
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
    use faer::Entity;
    use std::ops::Mul;

    // MatRef . Scalar -> Mat
    impl<E: Entity + Mul<E, Output = E>> ArgminDot<E, Mat<E>> for MatRef<'_, E> {
        #[inline]
        fn dot(&self, other: &E) -> Mat<E> {
            <Self as ArgminMul<E, _>>::mul(self, other)
        }
    }

    // Mat . Scalar -> Mat
    impl<E: Entity + Mul<E, Output = E>> ArgminDot<E, Mat<E>> for Mat<E> {
        #[inline]
        fn dot(&self, other: &E) -> Mat<E> {
            <_ as ArgminDot<E, _>>::dot(&self.as_mat_ref(), other)
        }
    }

    // MatRef . Scalar -> Mat
    impl<'a, E: Entity + Mul<E, Output = E>> ArgminDot<MatRef<'a, E>, Mat<E>> for E {
        #[inline]
        fn dot(&self, other: &MatRef<'a, E>) -> Mat<E> {
            <E as ArgminMul<MatRef<'a, E>, _>>::mul(self, other)
        }
    }

    // Mat . Scalar -> Mat
    impl<E: Entity + Mul<E, Output = E>> ArgminDot<Mat<E>, Mat<E>> for E {
        #[inline]
        fn dot(&self, other: &Mat<E>) -> Mat<E> {
            <E as ArgminDot<_, _>>::dot(self, &other.as_mat_ref())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::faer_tests::test_helper::*;
    use approx::assert_relative_eq;
    use faer::mat::AsMatRef;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_vec_vec_ $t>]() {
                    let a = vector3_new(1 as $t, 2 as $t, 3 as $t);
                    let b = vector3_new(4 as $t, 5 as $t, 6 as $t);
                    // all owned and reference type combinations
                    let res1: $t = <_ as ArgminDot<_, _>>::dot(&a, &b);
                    let res2: $t = <_ as ArgminDot<_, _>>::dot(&a.as_mat_ref(), &b);
                    let res3: $t = <_ as ArgminDot<_, _>>::dot(&a, &b.as_mat_ref());
                    let res4: $t = <_ as ArgminDot<_, _>>::dot(&a.as_mat_ref(), &b.as_mat_ref());
                    assert_relative_eq!(res1 as f64, 32 as f64, epsilon = f64::EPSILON);
                    assert_relative_eq!(res2 as f64, 32 as f64, epsilon = f64::EPSILON);
                    assert_relative_eq!(res3 as f64, 32 as f64, epsilon = f64::EPSILON);
                    assert_relative_eq!(res4 as f64, 32 as f64, epsilon = f64::EPSILON);
                }
            }

            item! {
                #[test]
                fn [<test_vec_scalar_ $t>]() {
                    let a = vector3_new(1 as $t, 2 as $t, 3 as $t);
                    let b = 2 as $t;
                    let product1: Mat<$t> =
                        <_ as ArgminDot<$t, _>>::dot(&a, &b);
                    let product2: Mat<$t> =
                        <_ as ArgminDot<$t, _>>::dot(&a.as_mat_ref(), &b);
                    let res = vector3_new(2 as $t, 4 as $t, 6 as $t);
                    assert_eq!(product1,product2);
                    assert_eq!(product1.nrows(),3);
                    assert_eq!(product1.ncols(),1);
                    for i in 0..3 {
                        assert_relative_eq!(res[(i,0)] as f64, product1[(i,0)] as f64, epsilon = f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_scalar_vec_ $t>]() {
                    let a = vector3_new(1 as $t, 2 as $t, 3 as $t);
                    let b = 2 as $t;
                    let product1: Mat<$t> =
                        <$t as ArgminDot<_, _>>::dot(&b, &a);
                    let product2: Mat<$t> =
                        <$t as ArgminDot<_, _>>::dot(&b, &a.as_mat_ref());
                    assert_eq!(product1,product2);
                    assert_eq!(product1.nrows(),3);
                    assert_eq!(product1.ncols(),1);
                    let res = vector3_new(2 as $t, 4 as $t, 6 as $t);
                    for i in 0..3 {
                        assert_relative_eq!(res[(i,0)] as f64, product1[(i,0)] as f64, epsilon = f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mat_vec_ $t>]() {
                    let a = vector3_new(1 as $t, 2 as $t, 3 as $t);
                    let b = row_vector3_new(4 as $t, 5 as $t, 6 as $t);
                    let res = matrix3_new(
                        4 as $t, 5 as $t, 6 as $t,
                        8 as $t, 10 as $t, 12 as $t,
                        12 as $t, 15 as $t, 18 as $t
                    );
                    let product1: Mat<$t> =
                        <_ as ArgminDot<_, _>>::dot(&a, &b);
                    let product2: Mat<$t> =
                        <_ as ArgminDot<_, _>>::dot(&a.as_mat_ref(), &b);
                    let product3: Mat<$t> =
                        <_ as ArgminDot<_, _>>::dot(&a, &b.as_mat_ref());
                    let product4: Mat<$t> =
                        <_ as ArgminDot<_, _>>::dot(&a.as_mat_ref(), &b.as_mat_ref());
                    for i in 0..3 {
                        for j in 0..3 {
                            assert_relative_eq!(res[(i, j)] as f64, product1[(i, j)] as f64, epsilon = f64::EPSILON);
                            assert_relative_eq!(res[(i, j)] as f64, product2[(i, j)] as f64, epsilon = f64::EPSILON);
                            assert_relative_eq!(res[(i, j)] as f64, product3[(i, j)] as f64, epsilon = f64::EPSILON);
                            assert_relative_eq!(res[(i, j)] as f64, product4[(i, j)] as f64, epsilon = f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mat_vec_2_ $t>]() {
                    let a = matrix3_new(
                        1 as $t, 2 as $t, 3 as $t,
                        4 as $t, 5 as $t, 6 as $t,
                        7 as $t, 8 as $t, 9 as $t
                    );
                    let b = vector3_new(1 as $t, 2 as $t, 3 as $t);
                    let res = vector3_new(14 as $t, 32 as $t, 50 as $t);
                    let product1: Mat<$t> =
                        <_ as ArgminDot<_, _>>::dot(&a, &b);
                    let product2: Mat<$t> =
                        <_ as ArgminDot<_, _>>::dot(&a.as_mat_ref(), &b);
                    let product3: Mat<$t> =
                        <_ as ArgminDot<_, _>>::dot(&a, &b.as_mat_ref());
                    let product4: Mat<$t> =
                        <_ as ArgminDot<_, _>>::dot(&a.as_mat_ref(), &b.as_mat_ref());
                    for i in 0..3 {
                        assert_relative_eq!(res[(i,0)] as f64, product1[(i,0)] as f64, epsilon = f64::EPSILON);
                        assert_relative_eq!(res[(i,0)] as f64, product2[(i,0)] as f64, epsilon = f64::EPSILON);
                        assert_relative_eq!(res[(i,0)] as f64, product3[(i,0)] as f64, epsilon = f64::EPSILON);
                        assert_relative_eq!(res[(i,0)] as f64, product4[(i,0)] as f64, epsilon = f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mat_mat_ $t>]() {
                    let a = matrix3_new(
                        1 as $t, 2 as $t, 3 as $t,
                        4 as $t, 5 as $t, 6 as $t,
                        3 as $t, 2 as $t, 1 as $t
                    );
                    let b = matrix3_new(
                        3 as $t, 2 as $t, 1 as $t,
                        6 as $t, 5 as $t, 4 as $t,
                        2 as $t, 4 as $t, 3 as $t
                    );
                    let res = matrix3_new(
                        21 as $t, 24 as $t, 18 as $t,
                        54 as $t, 57 as $t, 42 as $t,
                        23 as $t, 20 as $t, 14 as $t
                    );
                    let product1: Mat<$t> =
                        <_ as ArgminDot<_, _>>::dot(&a, &b);
                    let product2: Mat<$t> =
                        <_ as ArgminDot<_, _>>::dot(&a.as_mat_ref(), &b);
                    let product3: Mat<$t> =
                        <_ as ArgminDot<_, _>>::dot(&a, &b.as_mat_ref());
                    let product4: Mat<$t> =
                        <_ as ArgminDot<_, _>>::dot(&a.as_mat_ref(), &b.as_mat_ref());
                    for i in 0..3 {
                        for j in 0..3 {
                            assert_relative_eq!(res[(i, j)] as f64, product1[(i, j)] as f64, epsilon = f64::EPSILON);
                            assert_relative_eq!(res[(i, j)] as f64, product2[(i, j)] as f64, epsilon = f64::EPSILON);
                            assert_relative_eq!(res[(i, j)] as f64, product3[(i, j)] as f64, epsilon = f64::EPSILON);
                            assert_relative_eq!(res[(i, j)] as f64, product4[(i, j)] as f64, epsilon = f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mat_primitive_ $t>]() {
                    let a = matrix3_new(
                        1 as $t, 2 as $t, 3 as $t,
                        4 as $t, 5 as $t, 6 as $t,
                        3 as $t, 2 as $t, 1 as $t
                    );
                    let res = matrix3_new(
                        2 as $t, 4 as $t, 6 as $t,
                        8 as $t, 10 as $t, 12 as $t,
                        6 as $t, 4 as $t, 2 as $t
                    );
                    let product1: Mat<$t> =
                        <_ as ArgminDot<$t, _>>::dot(&a, &(2 as $t));
                    let product2: Mat<$t> =
                        <_ as ArgminDot<$t, _>>::dot(&a.as_mat_ref(), &(2 as $t));
                    assert_eq!(product1, product2);
                    assert_eq!(product1.nrows(), 3);
                    assert_eq!(product1.ncols(), 3);
                    for i in 0..3 {
                        for j in 0..3 {
                            assert_relative_eq!(res[(i, j)] as f64, product1[(i, j)] as f64, epsilon = f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_primitive_mat_ $t>]() {
                    let a = matrix3_new(
                        1 as $t, 2 as $t, 3 as $t,
                        4 as $t, 5 as $t, 6 as $t,
                        3 as $t, 2 as $t, 1 as $t
                    );
                    let res = matrix3_new(
                        2 as $t, 4 as $t, 6 as $t,
                        8 as $t, 10 as $t, 12 as $t,
                        6 as $t, 4 as $t, 2 as $t
                    );
                    let product1: Mat<$t> =
                        <$t as ArgminDot<_, _>>::dot(&(2 as $t), &a);
                    let product2: Mat<$t> =
                        <$t as ArgminDot<_, _>>::dot(&(2 as $t), &a.as_mat_ref());
                    assert_eq!(product1, product2);
                    assert_eq!(product1.nrows(), 3);
                    assert_eq!(product1.ncols(), 3);
                    for i in 0..3 {
                        for j in 0..3 {
                            assert_relative_eq!(res[(i, j)] as f64, product1[(i, j)] as f64, epsilon = f64::EPSILON);
                        }
                    }
                }
            }
        };
    }

    make_test!(f32);
    make_test!(f64);
}
