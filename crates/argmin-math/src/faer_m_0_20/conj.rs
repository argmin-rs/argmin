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
        // on the MatRef struct since it's not guaranteed to return matrix same type.
        // Thus, we implement the conjugation using the num-complex trait manually
        faer::zipped_rw!(self).map(|unzipped!(this)| ComplexFloat::conj(this.read()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::faer_tests::test_helper::*;
    use approx::assert_relative_eq;
    use faer::linalg::entity::complex_split::ComplexConj;
    use num_complex::Complex;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_conj_complex_faer_ $t>]() {
                        let a : Mat<Complex<$t>> = vector3_new(
                        Complex::new(1 as $t, 2 as $t),
                        Complex::new(4 as $t, -3 as $t),
                        Complex::new(8 as $t, 0 as $t)
                    );
                    let b = vector3_new(
                        Complex::new(1 as $t, -2 as $t),
                        Complex::new(4 as $t, 3 as $t),
                        Complex::new(8 as $t, 0 as $t)
                    );
                    let res: Mat<_> = <Mat<Complex<$t>> as ArgminConj>::conj(&a);
                    assert_eq!(res.nrows(),3);
                    assert_eq!(res.ncols(),1);
                    for i in 0..3 {
                        assert_relative_eq!(b.read(i,0).re(), res.read(i,0).re(), epsilon = $t::EPSILON);
                        assert_relative_eq!(b.read(i,0).im(), res.read(i,0).im(), epsilon = $t::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_conj_faer_ $t>]() {
                    let a = vector3_new(1 as $t, 4 as $t, 8 as $t);
                    let b = vector3_new(1 as $t, 4 as $t, 8 as $t);
                    let res = <_ as ArgminConj>::conj(&a);
                    for i in 0..3 {
                        assert_relative_eq!(b[(i,0)], res[(i,0)], epsilon = $t::EPSILON);
                    }
                }
            }
        };
    }

    make_test!(f32);
    make_test!(f64);
}
