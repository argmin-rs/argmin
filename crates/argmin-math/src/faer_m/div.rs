use super::RealEntity;
use crate::ArgminDiv;
use faer::{
    mat::AsMatRef,
    reborrow::{IntoConst, Reborrow, ReborrowMut},
    Conjugate, Mat, MatMut, MatRef, SimpleEntity,
};
use std::ops::DivAssign;

impl<'a, E> ArgminDiv<E, Mat<E::Canonical>> for MatRef<'a, E>
where
    E: RealEntity,
    E::Canonical: DivAssign<E>,
{
    fn div(&self, other: &E) -> Mat<E::Canonical> {
        let mut owned = MatRef::to_owned(self);
        owned
            .col_iter_mut()
            .flat_map(|col_iter| col_iter.iter_mut())
            .for_each(|elem| elem.div_assign(*other));

        owned
    }
}

impl<E> ArgminDiv<E, Mat<E::Canonical>> for Mat<E>
where
    E: RealEntity,
    E::Canonical: DivAssign<E>,
{
    #[inline]
    fn div(&self, other: &E) -> Mat<E::Canonical> {
        <MatRef<E> as ArgminDiv<_, _>>::div(&self.as_mat_ref(), other)
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn more_tests() {
        todo!()
    }
}
