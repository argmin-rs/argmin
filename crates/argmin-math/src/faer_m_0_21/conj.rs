use crate::ArgminConj;
use faer::{mat::AsMatRef, reborrow::ReborrowMut, unzip, Mat, MatMut, MatRef};
use faer_traits::ComplexField;
use num_complex::ComplexFloat;

impl<E: ComplexField + num_complex::ComplexFloat> ArgminConj for Mat<E> {
    #[inline]
    fn conj(&self) -> Self {
        //@note(geo-ant): we can't directly use the `conjugate()' function
        // on the MatRef struct since it's not guaranteed to return a
        // matrix of the same type. Thus, we implement the conjugation
        // using the num-complex trait manually.
        faer::zip!(self).map(|unzip!(this)| this.conj())
    }
}
