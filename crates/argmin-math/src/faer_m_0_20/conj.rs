use crate::ArgminConj;
use faer::{
    mat::AsMatRef, reborrow::ReborrowMut, unzipped, Conjugate, Entity, Mat, MatMut, MatRef,
    SimpleEntity,
};
use num_complex::ComplexFloat;

impl<E: Entity + num_complex::ComplexFloat> ArgminConj for Mat<E> {
    #[inline]
    fn conj(&self) -> Self {
        //@note(geo-ant): we can't directly use the `conjugate()' function
        // on the MatRef struct since it's not guaranteed to return a
        // matrix of the same type. Thus, we implement the conjugation
        // using the num-complex trait manually.
        faer::zipped_rw!(self).map(|unzipped!(this)| ComplexFloat::conj(this.read()))
    }
}
