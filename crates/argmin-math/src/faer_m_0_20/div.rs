use crate::ArgminDiv;
use faer::{
    mat::AsMatRef,
    reborrow::{IntoConst, Reborrow, ReborrowMut},
    unzipped, zipped_rw, Conjugate, Entity, Mat, MatMut, MatRef, SimpleEntity,
};
use std::ops::{Div, DivAssign};

/// MatRef / Scalar -> MatRef
impl<E> ArgminDiv<E, Mat<E>> for MatRef<'_, E>
where
    E: Entity + Div<E, Output = E>,
{
    #[inline]
    fn div(&self, other: &E) -> Mat<E> {
        zipped_rw!(self).map(|unzipped!(this)| this.read() / *other)
    }
}

/// Mat / Scalar -> Mat
impl<E> ArgminDiv<E, Mat<E>> for Mat<E>
where
    E: Entity + Div<E, Output = E>,
{
    #[inline]
    fn div(&self, other: &E) -> Mat<E> {
        //@note(geo-ant) because we are taking self by reference we
        // cannot mutate the matrix in place, so we can just as well
        // reuse the reference code
        <_ as ArgminDiv<_, _>>::div(&self.as_mat_ref(), other)
    }
}

/// Scalar / MatRef -> Mat
impl<'a, E> ArgminDiv<MatRef<'a, E>, Mat<E>> for E
where
    E: Entity + Div<E, Output = E>,
{
    #[inline]
    fn div(&self, other: &MatRef<'a, E>) -> Mat<E> {
        // does not commute with the expressions above, which is why
        // we need our own implementations
        zipped_rw!(other).map(|unzipped!(other_elem)| *self / other_elem.read())
    }
}

/// Scalar / Mat -> Mat
impl<E> ArgminDiv<Mat<E>, Mat<E>> for E
where
    E: Entity + Div<E, Output = E>,
{
    #[inline]
    fn div(&self, other: &Mat<E>) -> Mat<E> {
        //@note(geo-ant) because we are taking self by reference we
        // cannot mutate the matrix in place, so we can just as well
        // reuse the reference code
        <_ as ArgminDiv<_, _>>::div(self, &other.as_mat_ref())
    }
}

/// MatRef / MatRef -> Mat (pointwise division)
impl<'a, E: Entity + Div<E, Output = E>> ArgminDiv<MatRef<'a, E>, Mat<E>> for MatRef<'_, E> {
    #[inline]
    fn div(&self, other: &MatRef<'a, E>) -> Mat<E> {
        zipped_rw!(self, other).map(|unzipped!(this, other)| this.read() / other.read())
    }
}

/// Mat / MatRef -> Mat (pointwise division)
impl<'a, E: Entity + Div<E, Output = E>> ArgminDiv<MatRef<'a, E>, Mat<E>> for Mat<E> {
    #[inline]
    fn div(&self, other: &MatRef<'a, E>) -> Mat<E> {
        <_ as ArgminDiv<_, _>>::div(&self.as_mat_ref(), other)
    }
}

/// MatRef / Mat-> Mat (pointwise division)
impl<E: Entity + Div<E, Output = E>> ArgminDiv<Mat<E>, Mat<E>> for MatRef<'_, E> {
    #[inline]
    fn div(&self, other: &Mat<E>) -> Mat<E> {
        <_ as ArgminDiv<_, _>>::div(self, &other.as_mat_ref())
    }
}

/// Mat / Mat-> Mat (pointwise division)
impl<E: Entity + Div<E, Output = E>> ArgminDiv<Mat<E>, Mat<E>> for Mat<E> {
    #[inline]
    fn div(&self, other: &Mat<E>) -> Mat<E> {
        <_ as ArgminDiv<_, _>>::div(&self.as_mat_ref(), &other.as_mat_ref())
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
                fn [<test_div_vec_scalar_ $t>]() {
                    let a = vector3_new(4 as $t, 16 as $t, 8 as $t);
                    let b = 2 as $t;
                    let target = vector3_new(2 as $t, 8 as $t, 4 as $t);
                    let res1 = <_ as ArgminDiv<$t, _>>::div(&a, &b);
                    let res2 = <_ as ArgminDiv<$t, _>>::div(&a.as_mat_ref(), &b);
                    assert_eq!(res1,res2);
                    assert_eq!(res1.nrows(),3);
                    assert_eq!(res1.ncols(),1);
                    for i in 0..3 {
                        assert_relative_eq!(target[(i,0)] as f64, res1[(i,0)] as f64, epsilon = f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_div_scalar_vec_ $t>]() {
                    let a = vector3_new(2 as $t, 4 as $t, 8 as $t);
                    let b = 32 as $t;
                    let target = vector3_new(16 as $t, 8 as $t, 4 as $t);
                    let res1 = <$t as ArgminDiv<_, _>>::div(&b, &a);
                    let res2 = <$t as ArgminDiv<_, _>>::div(&b, &a);
                    assert_eq!(res1,res2);
                    assert_eq!(res1.nrows(),3);
                    assert_eq!(res1.ncols(),1);
                    for i in 0..3 {
                        assert_relative_eq!(target[(i,0)] as f64, res1[(i,0)] as f64, epsilon = f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_div_vec_vec_ $t>]() {
                    let a = vector3_new(4 as $t, 9 as $t, 8 as $t);
                    let b = vector3_new(2 as $t, 3 as $t, 4 as $t);
                    let target = vector3_new(2 as $t, 3 as $t, 2 as $t);
                    let res1 :Mat<$t> = <_ as ArgminDiv<_, _>>::div(&a, &b);
                    let res2 :Mat<$t> = <_ as ArgminDiv<_, _>>::div(&a, &b);
                    let res3 :Mat<$t> = <_ as ArgminDiv<_, _>>::div(&a, &b);
                    let res4 :Mat<$t> = <_ as ArgminDiv<_, _>>::div(&a, &b);
                    assert_eq!(res1,res2);
                    assert_eq!(res1,res3);
                    assert_eq!(res1,res4);
                    assert_eq!(res1.nrows(),3);
                    assert_eq!(res1.ncols(),1);
                    for i in 0..3 {
                        assert_relative_eq!(target[(i,0)] as f64, res1[(i,0)] as f64, epsilon = f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_div_vec_vec_panic_ $t>]() {
                    let a = column_vector_from_vec(vec![1 as $t, 4 as $t]);
                    let b = column_vector_from_vec(vec![41 as $t, 38 as $t, 34 as $t]);
                    <_ as ArgminDiv<_, _>>::div(&a, &b);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_div_vec_vec_panic_2_ $t>]() {
                    let a = column_vector_from_vec(vec![]); let b = column_vector_from_vec(vec![41 as $t, 38 as $t, 34 as $t]); <_ as ArgminDiv<_, _>>::div(&a, &b);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_div_vec_vec_panic_3_ $t>]() {
                    let a = column_vector_from_vec(vec![41 as $t, 38 as $t, 34 as $t]);
                    let b = column_vector_from_vec(vec![]);
                    <_ as ArgminDiv<_,_>>::div(&a, &b);
                }
            }

            item! {
                #[test]
                fn [<test_div_mat_mat_ $t>]() {
                    let a = matrix2x3_new(
                        4 as $t, 12 as $t, 8 as $t,
                        9 as $t, 20 as $t, 45 as $t
                    );
                    let b = matrix2x3_new(
                        2 as $t, 3 as $t, 4 as $t,
                        3 as $t, 4 as $t, 5 as $t
                    );
                    let target = matrix2x3_new(
                        2 as $t, 4 as $t, 2 as $t,
                        3 as $t, 5 as $t, 9 as $t
                    );
                    let res1 = <_ as ArgminDiv<_, _>>::div(&a, &b);
                    let res2 = <_ as ArgminDiv<_, _>>::div(&a, &b);
                    let res3 = <_ as ArgminDiv<_, _>>::div(&a, &b);
                    let res4 = <_ as ArgminDiv<_, _>>::div(&a, &b);
                    assert_eq!(res1,res2);
                    assert_eq!(res1,res3);
                    assert_eq!(res1,res4);
                    assert_eq!(res1.nrows(),2);
                    assert_eq!(res1.ncols(),3);
                    for i in 0..3 {
                        for j in 0..2 {
                            assert_relative_eq!(target[(j, i)] as f64, res1[(j, i)] as f64, epsilon = f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_div_mat_mat_panic_2_ $t>]() {
                    let a = matrix2x3_new(
                        1 as $t, 4 as $t, 8 as $t,
                        2 as $t, 5 as $t, 9 as $t
                    );
                    let b = faer::mat![
                        [41 as $t, 38 as $t]
                    ];
                    <_ as ArgminDiv<_, _>>::div(&a, &b);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_div_mat_mat_panic_3_ $t>]() {
                    let a = matrix2x3_new(
                        1 as $t, 4 as $t, 8 as $t,
                        2 as $t, 5 as $t, 9 as $t
                    );
                    let b = Mat::new();
                    <_ as ArgminDiv<_, _>>::div(&a, &b);
                }
            }
        };
    }

    make_test!(f32);
    make_test!(f64);
}
