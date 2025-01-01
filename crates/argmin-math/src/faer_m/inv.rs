use crate::ArgminInv;
use faer::{
    dyn_stack::{GlobalPodBuffer, PodStack},
    linalg::lu::partial_pivoting::compute::{lu_in_place, lu_in_place_req},
    mat::{AsMatMut, AsMatRef},
    prelude::SolverCore,
    reborrow::ReborrowMut,
    ComplexField, Entity, Mat, MatRef, SimpleEntity,
};
use std::fmt;

#[derive(Debug, thiserror::Error, PartialEq)]
struct InverseError;

impl fmt::Display for InverseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Non-invertible matrix")
    }
}

/// calculate the inverse via LU decomposition with partial pivoting
impl<E: Entity + ComplexField> ArgminInv<Mat<E>> for MatRef<'_, E> {
    #[inline]
    fn inv(&self) -> Result<Mat<E>, anyhow::Error> {
        if self.nrows() != self.ncols() || self.nrows() == 0 {
            Err(InverseError.into())
        } else {
            Ok(self.partial_piv_lu().inverse())
        }
    }
}

/// calculate the inverse via LU decomposition with partial pivoting
impl<E: Entity + ComplexField> ArgminInv<Mat<E>> for Mat<E> {
    #[inline]
    fn inv(&self) -> Result<Mat<E>, anyhow::Error> {
        if self.nrows() != self.ncols() || self.nrows() == 0 {
            Err(InverseError.into())
        } else {
            Ok(self.partial_piv_lu().inverse())
        }
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
                fn [<test_inv_ $t>]() {
                    let a = matrix2_new(
                        2 as $t, 5 as $t,
                        1 as $t, 3 as $t,
                    );
                    let target = matrix2_new(
                        3 as $t, -5 as $t,
                        -1 as $t, 2 as $t,
                    );
                    let res = <_ as ArgminInv<_>>::inv(&a).unwrap();
                    let res1 = <_ as ArgminInv<_>>::inv(&a.as_mat_ref()).unwrap();
                    assert_eq!(res,res1);
                    assert_eq!(res.nrows(),2);
                    assert_eq!(res.ncols(),2);
                    for i in 0..2 {
                        for j in 0..2 {
                            assert_relative_eq!(res[(i, j)], target[(i, j)], epsilon = $t::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_inv_error $t>]() {
                    let a = matrix2_new(
                        2 as $t, 5 as $t,
                        4 as $t, 10 as $t,
                    );
                    let err = <_ as ArgminInv<_>>::inv(&a).unwrap_err().downcast::<InverseError>().unwrap();
                    assert_eq!(err, InverseError {});
                }
            }
        };
    }

    make_test!(f32);
    make_test!(f64);
}
