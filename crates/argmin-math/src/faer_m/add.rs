use std::ops::AddAssign;

use crate::ArgminAdd;
use faer::{mat::AsMatRef, Conjugate, Mat, MatRef, SimpleEntity};

impl<'a, E> ArgminAdd<E, Mat<E::Canonical>> for MatRef<'a, E>
where
    E: SimpleEntity + Conjugate<Canonical = E>,
    E::Canonical: AddAssign<E>,
{
    fn add(&self, other: &E) -> Mat<E::Canonical> {
        let mut owned = MatRef::to_owned(self);
        owned
            .col_iter_mut()
            .flat_map(|col_iter| col_iter.iter_mut())
            .for_each(|elem| elem.add_assign(*other));

        owned
    }
}

#[test]
fn test_blah() {
    fn add_scalar(scalar: f64, mat: &Mat<f64>) -> Mat<f64> {
        <MatRef<f64> as ArgminAdd<f64, Mat<f64>>>::add(&mat.as_mat_ref(), &scalar)
    }
    let mat = Mat::<f64>::zeros(10, 11);
    let mut expected = Mat::<f64>::zeros(10, 11);
    let mat = add_scalar(1., &mat);
    expected.fill(1.);
    assert_eq!(mat, expected);
}
