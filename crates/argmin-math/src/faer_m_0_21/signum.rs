use crate::ArgminSignum;
use faer::{Mat, MatRef, Shape};
use faer_traits::ComplexField;
use num_complex::Complex;

/// helper trait that indicates the signum of a numeric value can
/// be calculated
//@note(geo-ant): one could also decide to implement ArgminSignum
// on the primitive types themselves.
trait SignumInternal {
    fn signum_internal(&self) -> Self;
}

macro_rules! make_signum {
    ($t:ty) => {
        impl SignumInternal for $t {
            fn signum_internal(&self) -> Self {
                self.signum()
            }
        }
    };
}

macro_rules! make_signum_complex {
    ($t:ty) => {
        impl SignumInternal for $t {
            fn signum_internal(&self) -> Self {
                Complex {
                    re: self.re.signum(),
                    im: self.im.signum(),
                }
            }
        }
    };
}

make_signum!(i8);
make_signum!(i16);
make_signum!(i32);
make_signum!(i64);
make_signum!(f32);
make_signum!(f64);
make_signum_complex!(Complex<i8>);
make_signum_complex!(Complex<i16>);
make_signum_complex!(Complex<i32>);
make_signum_complex!(Complex<i64>);
make_signum_complex!(Complex<f32>);
make_signum_complex!(Complex<f64>);

impl<E, R, C> ArgminSignum for Mat<E, R, C>
where
    E: ComplexField + SignumInternal,
    R: Shape,
    C: Shape,
{
    #[inline]
    fn signum(self) -> Self {
        faer::zip!(&self).map(|faer::unzip!(elem)| elem.signum_internal())
    }
}
