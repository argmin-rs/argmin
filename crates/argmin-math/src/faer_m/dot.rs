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
mod tests {
    use super::super::test_helper::*;
    use super::*;
    use approx::assert_relative_eq;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_vec_vec_ $t>]() {
                    let a = vector3_new(1 as $t, 2 as $t, 3 as $t);
                    let b = vector3_new(4 as $t, 5 as $t, 6 as $t);
                    let res: $t = <Vector3<$t> as ArgminDot<Vector3<$t>, $t>>::dot(&a, &b);
                    assert_relative_eq!(res as f64, 32 as f64, epsilon = f64::EPSILON);
                }
            }

            item! {
                #[test]
                fn [<test_vec_scalar_ $t>]() {
                    let a = Vector3::new(1 as $t, 2 as $t, 3 as $t);
                    let b = 2 as $t;
                    let product: Vector3<$t> =
                        <Vector3<$t> as ArgminDot<$t, Vector3<$t>>>::dot(&a, &b);
                    let res = Vector3::new(2 as $t, 4 as $t, 6 as $t);
                    for i in 0..3 {
                        assert_relative_eq!(res[i] as f64, product[i] as f64, epsilon = f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_scalar_vec_ $t>]() {
                    let a = Vector3::new(1 as $t, 2 as $t, 3 as $t);
                    let b = 2 as $t;
                    let product: Vector3<$t> =
                        <$t as ArgminDot<Vector3<$t>, Vector3<$t>>>::dot(&b, &a);
                    let res = Vector3::new(2 as $t, 4 as $t, 6 as $t);
                    for i in 0..3 {
                        assert_relative_eq!(res[i] as f64, product[i] as f64, epsilon = f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mat_vec_ $t>]() {
                    let a = Vector3::new(1 as $t, 2 as $t, 3 as $t);
                    let b = RowVector3::new(4 as $t, 5 as $t, 6 as $t);
                    let res = Matrix3::new(
                        4 as $t, 5 as $t, 6 as $t,
                        8 as $t, 10 as $t, 12 as $t,
                        12 as $t, 15 as $t, 18 as $t
                    );
                    let product: Matrix3<$t> =
                        <Vector3<$t> as ArgminDot<RowVector3<$t>, Matrix3<$t>>>::dot(&a, &b);
                    for i in 0..3 {
                        for j in 0..3 {
                            assert_relative_eq!(res[(i, j)] as f64, product[(i, j)] as f64, epsilon = f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mat_vec_2_ $t>]() {
                    let a = Matrix3::new(
                        1 as $t, 2 as $t, 3 as $t,
                        4 as $t, 5 as $t, 6 as $t,
                        7 as $t, 8 as $t, 9 as $t
                    );
                    let b = Vector3::new(1 as $t, 2 as $t, 3 as $t);
                    let res = Vector3::new(14 as $t, 32 as $t, 50 as $t);
                    let product: Vector3<$t> =
                        <Matrix3<$t> as ArgminDot<Vector3<$t>, Vector3<$t>>>::dot(&a, &b);
                    for i in 0..3 {
                        assert_relative_eq!(res[i] as f64, product[i] as f64, epsilon = f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mat_mat_ $t>]() {
                    let a = Matrix3::new(
                        1 as $t, 2 as $t, 3 as $t,
                        4 as $t, 5 as $t, 6 as $t,
                        3 as $t, 2 as $t, 1 as $t
                    );
                    let b = Matrix3::new(
                        3 as $t, 2 as $t, 1 as $t,
                        6 as $t, 5 as $t, 4 as $t,
                        2 as $t, 4 as $t, 3 as $t
                    );
                    let res = Matrix3::new(
                        21 as $t, 24 as $t, 18 as $t,
                        54 as $t, 57 as $t, 42 as $t,
                        23 as $t, 20 as $t, 14 as $t
                    );
                    let product: Matrix3<$t> =
                        <Matrix3<$t> as ArgminDot<Matrix3<$t>, Matrix3<$t>>>::dot(&a, &b);
                    for i in 0..3 {
                        for j in 0..3 {
                            assert_relative_eq!(res[(i, j)] as f64, product[(i, j)] as f64, epsilon = f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mat_primitive_ $t>]() {
                    let a = Matrix3::new(
                        1 as $t, 2 as $t, 3 as $t,
                        4 as $t, 5 as $t, 6 as $t,
                        3 as $t, 2 as $t, 1 as $t
                    );
                    let res = Matrix3::new(
                        2 as $t, 4 as $t, 6 as $t,
                        8 as $t, 10 as $t, 12 as $t,
                        6 as $t, 4 as $t, 2 as $t
                    );
                    let product: Matrix3<$t> =
                        <Matrix3<$t> as ArgminDot<$t, Matrix3<$t>>>::dot(&a, &(2 as $t));
                    for i in 0..3 {
                        for j in 0..3 {
                            assert_relative_eq!(res[(i, j)] as f64, product[(i, j)] as f64, epsilon = f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_primitive_mat_ $t>]() {
                    let a = Matrix3::new(
                        1 as $t, 2 as $t, 3 as $t,
                        4 as $t, 5 as $t, 6 as $t,
                        3 as $t, 2 as $t, 1 as $t
                    );
                    let res = Matrix3::new(
                        2 as $t, 4 as $t, 6 as $t,
                        8 as $t, 10 as $t, 12 as $t,
                        6 as $t, 4 as $t, 2 as $t
                    );
                    let product: Matrix3<$t> =
                        <$t as ArgminDot<Matrix3<$t>, Matrix3<$t>>>::dot(&(2 as $t), &a);
                    for i in 0..3 {
                        for j in 0..3 {
                            assert_relative_eq!(res[(i, j)] as f64, product[(i, j)] as f64, epsilon = f64::EPSILON);
                        }
                    }
                }
            }
        };
    }

    make_test!(f32);
    make_test!(f64);
}
