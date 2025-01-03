use crate::ArgminAdd;
use faer::{
    mat::{AsMatMut, AsMatRef},
    reborrow::{IntoConst, Reborrow, ReborrowMut},
    unzipped, zipped, zipped_rw, ComplexField, Conjugate, Entity, Mat, MatMut, MatRef,
    SimpleEntity,
};
use std::ops::{Add, AddAssign};

/// MatRef + Scalar -> Mat
impl<E, R, C> ArgminAdd<E, Mat<E, R, C>> for MatRef<'_, E, R, C>
where
    E: Entity + Add<E, Output = E>,
    R: faer::Shape,
    C: faer::Shape,
{
    #[inline]
    fn add(&self, other: &E) -> Mat<E, R, C> {
        zipped_rw!(self).map(|unzipped!(this)| this.read() + *other)
    }
}

/// Scaler + MatRef-> Mat
impl<'a, E, R, C> ArgminAdd<MatRef<'a, E, R, C>, Mat<E, R, C>> for E
where
    E: Entity + Add<E, Output = E>,
    R: faer::Shape,
    C: faer::Shape,
{
    #[inline]
    fn add(&self, other: &MatRef<'a, E, R, C>) -> Mat<E, R, C> {
        // commutative with MatRef + Scalar so we can fall back on that case
        <_ as ArgminAdd<_, _>>::add(other, self)
    }
}

//@todo(geo) also add scalar + Matrix and matrix + Scalar (and reference variants?)

/// Mat + Scalar -> Mat
impl<E, R, C> ArgminAdd<E, Mat<E, R, C>> for Mat<E, R, C>
where
    E: Entity + Add<E, Output = E>,
    R: faer::Shape,
    C: faer::Shape,
{
    #[inline]
    fn add(&self, other: &E) -> Mat<E, R, C> {
        //@note(geo-ant) because we are taking self by reference we
        // cannot mutate the matrix in place, so we can just as well
        // reuse the reference code
        <_ as ArgminAdd<_, _>>::add(&self.as_mat_ref(), other)
    }
}

/// Scalar + Mat -> Mat
impl<E, R, C> ArgminAdd<Mat<E, R, C>, Mat<E, R, C>> for E
where
    E: Entity + Add<E, Output = E>,
    R: faer::Shape,
    C: faer::Shape,
{
    #[inline]
    fn add(&self, other: &Mat<E, R, C>) -> Mat<E, R, C> {
        // commutative with Mat + Scalar so we can fall back on that case
        <_ as ArgminAdd<_, _>>::add(other, self)
    }
}

/// MatRef + MatRef -> Mat
impl<'a, E> ArgminAdd<MatRef<'a, E>, Mat<E>> for MatRef<'_, E>
where
    E: Entity + ComplexField,
{
    #[inline]
    fn add(&self, other: &MatRef<'a, E>) -> Mat<E> {
        <_ as Add>::add(self, other)
    }
}

/// MatRef + Mat -> Mat
impl<E: Entity + ComplexField> ArgminAdd<Mat<E>, Mat<E>> for MatRef<'_, E> {
    #[inline]
    fn add(&self, other: &Mat<E>) -> Mat<E> {
        self + other
    }
}

/// Mat + Mat -> Mat
impl<E: Entity + ComplexField> ArgminAdd<Mat<E>, Mat<E>> for Mat<E> {
    #[inline]
    fn add(&self, other: &Mat<E>) -> Mat<E> {
        self + other
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helper::*;
    use super::*;
    use approx::assert_relative_eq;
    use faer::mat::AsMatRef;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_add_vec_scalar_ $t>]() {
                    let a = vector3_new(1 as $t, 4 as $t, 8 as $t);
                    let b = 34 as $t;
                    let target = vector3_new(35 as $t, 38 as $t, 42 as $t);
                    let res = <_ as ArgminAdd<$t, _>>::add(&a, &b);
                    for i in 0..3 {
                        assert_relative_eq!(target[(i,0)] as f64, res[(i,0)] as f64, epsilon = f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_add_scalar_vec_ $t>]() {
                    let a = vector3_new(1 as $t, 4 as $t, 8 as $t);
                    let b = 34 as $t;
                    let target = vector3_new(35 as $t, 38 as $t, 42 as $t);
                    let res = <$t as ArgminAdd<_,_>>::add(&b, &a);
                    for i in 0..3 {
                        assert_relative_eq!(target[(i,0)] as f64, res[(i,0)] as f64, epsilon = f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_add_vec_vec_ $t>]() {
                    let a = vector3_new(1 as $t, 4 as $t, 8 as $t);
                    let b = vector3_new(41 as $t, 38 as $t, 34 as $t);
                    let target = vector3_new(42 as $t, 42 as $t, 42 as $t);
                    let res = <_ as ArgminAdd<_, _>>::add(&a, &b);
                    for i in 0..3 {
                        assert_relative_eq!(target[(i,0)] as f64, res[(i,0)] as f64, epsilon = f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_add_vec_vec_panic_ $t>]() {
                    let a = column_vector_from_vec(vec![1 as $t, 4 as $t]);
                    let b = column_vector_from_vec(vec![41 as $t, 38 as $t, 34 as $t]);
                    <_ as ArgminAdd<_,_>>::add(&a, &b);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_add_vec_vec_panic_2_ $t>]() {
                    let a = column_vector_from_vec(vec![]);
                    let b = column_vector_from_vec(vec![41 as $t, 38 as $t, 34 as $t]);
                    <_ as ArgminAdd<_, _>>::add(&a, &b);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_add_vec_vec_panic_3_ $t>]() {
                    let a = column_vector_from_vec(vec![41 as $t, 38 as $t, 34 as $t]);
                    let b = column_vector_from_vec(vec![]);
                    <_ as ArgminAdd<_, _>>::add(&a, &b);
                }
            }

            item! {
                #[test]
                fn [<test_add_mat_mat_ $t>]() {
                    let a = matrix2x3_new(
                        1 as $t, 4 as $t, 8 as $t,
                        2 as $t, 5 as $t, 9 as $t
                    );
                    let b = matrix2x3_new(
                        41 as $t, 38 as $t, 34 as $t,
                        40 as $t, 37 as $t, 33 as $t
                    );
                    let target = matrix2x3_new(
                        42 as $t, 42 as $t, 42 as $t,
                        42 as $t, 42 as $t, 42 as $t
                    );
                    let res = <_ as ArgminAdd<_, _>>::add(&a, &b);
                    for i in 0..3 {
                        for j in 0..2 {
                            assert_relative_eq!(target[(j, i)] as f64, res[(j, i)] as f64, epsilon = f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_add_mat_scalar_ $t>]() {
                    let a = matrix2x3_new(
                        1 as $t, 4 as $t, 8 as $t,
                        2 as $t, 5 as $t, 9 as $t
                    );
                    let b = 2 as $t;
                    let target = matrix2x3_new(
                        3 as $t, 6 as $t, 10 as $t,
                        4 as $t, 7 as $t, 11 as $t
                    );
                    let res = <_ as ArgminAdd<$t, _>>::add(&a, &b);
                    for i in 0..3 {
                        for j in 0..2 {
                            assert_relative_eq!(target[(j, i)] as f64, res[(j, i)] as f64, epsilon = f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_add_mat_mat_panic_2_ $t>]() {
                    let a = faer::mat![
                        [1 as $t, 4 as $t, 8 as $t],
                        [2 as $t, 5 as $t, 9 as $t]
                    ];
                    let b = faer::mat![
                        [41 as $t, 38 as $t]
                    ];
                    <_ as ArgminAdd<_, _>>::add(&a, &b);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_add_mat_mat_panic_3_ $t>]() {
                    let a = faer::mat![
                        [1 as $t, 4 as $t, 8 as $t],
                        [2 as $t, 5 as $t, 9 as $t]
                    ];
                    let b = faer::Mat::new();
                    <_ as ArgminAdd<_, _>>::add(&a, &b);
                }
            }
        };
    }

    make_test!(f32);
    make_test!(f64);
}
