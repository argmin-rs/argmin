use super::RealEntity;
use crate::ArgminDiv;
use faer::{
    mat::AsMatRef,
    reborrow::{IntoConst, Reborrow, ReborrowMut},
    unzipped, zipped_rw, Conjugate, Entity, Mat, MatMut, MatRef, SimpleEntity,
};
use std::ops::{Div, DivAssign};

/// MatRef / Scalar -> MatRef
impl<'a, E> ArgminDiv<E, Mat<E>> for MatRef<'a, E>
where
    E: Entity + Div<E, Output = E>,
{
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
        faer::zipped_rw!(self).map(|unzipped!(this)| this.read() / *other)
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn more_tests() {
        todo!()
    }
}
