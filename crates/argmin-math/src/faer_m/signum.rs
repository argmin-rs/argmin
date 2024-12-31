use crate::ArgminSignum;
use faer::{unzipped, zipped_rw, Entity, Mat, MatRef, Shape};
use num_complex::Complex;

/// helper trait that indicates the signum of a numeric value can
/// be calculated
//@note(geo-ant): one could also decide to implement ArgminSignum
// on the primitive types themselves.
trait SignumInternal {
    fn signum_internal(self) -> Self;
}

macro_rules! make_signum {
    ($t:ty) => {
        impl SignumInternal for $t {
            fn signum_internal(self) -> Self {
                self.signum()
            }
        }
    };
}

macro_rules! make_signum_complex {
    ($t:ty) => {
        impl SignumInternal for $t {
            fn signum_internal(self) -> Self {
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
    E: Entity + SignumInternal,
    R: Shape,
    C: Shape,
{
    #[inline]
    fn signum(self) -> Self {
        zipped_rw!(self).map(|unzipped!(elem)| elem.read().signum_internal())
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helper::*;
    use super::*;
    use approx::assert_relative_eq;
    use faer::mat;
    use faer::Mat;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_signum_ $t>]() {
                    let a = column_vector_from_vec(vec![3 as $t, -4 as $t, -8 as $t]);
                    let b = column_vector_from_vec(vec![1 as $t, -1 as $t, -1 as $t]);
                    let res = <_ as ArgminSignum>::signum(a);
                    for i in 0..3 {
                        assert_relative_eq!(b[(i,0)], res[(i,0)], epsilon = $t::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_signum_scalar_mat_2_ $t>]() {
                    let b = mat![
                        [3 as $t, -4 as $t, 8 as $t],
                        [-2 as $t, -5 as $t, 9 as $t]
                    ];
                    let target = mat![
                        [1 as $t, -1 as $t, 1 as $t],
                        [-1 as $t, -1 as $t, 1 as $t]
                    ];
                    let res = b.signum();
                    for i in 0..3 {
                        for j in 0..2 {
                            assert_relative_eq!(target[(j, i)], res[(j, i)], epsilon = $t::EPSILON);
                        }
                    }
                }
            }
        };
    }

    make_test!(f32);
    make_test!(f64);
}
