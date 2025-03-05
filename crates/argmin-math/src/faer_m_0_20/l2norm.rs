use crate::ArgminL2Norm;
use faer::{ComplexField, Entity, Mat, MatRef, SimpleEntity};

impl<E: Entity + ComplexField> ArgminL2Norm<E::Real> for MatRef<'_, E> {
    fn l2_norm(&self) -> E::Real {
        self.norm_l2()
    }
}

impl<E: Entity + ComplexField> ArgminL2Norm<E::Real> for Mat<E> {
    fn l2_norm(&self) -> E::Real {
        self.norm_l2()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::faer_tests::test_helper::*;
    use approx::assert_relative_eq;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_norm_ $t>]() {
                    let a = vector2_new(4 as $t, 3 as $t);
                    let res = <_ as ArgminL2Norm<$t>>::l2_norm(&a);
                    let target = 5 as $t;
                    assert_relative_eq!(target as $t, res as $t, epsilon = $t::EPSILON);
                }
            }
        };
    }

    macro_rules! make_test_signed {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_norm_signed_ $t>]() {
                    let a = vector2_new(-4 as $t, -3 as $t);
                    let res = <_ as ArgminL2Norm<$t>>::l2_norm(&a);
                    let target = 5 as $t;
                    assert_relative_eq!(target as $t, res as $t, epsilon = $t::EPSILON);
                }
            }
        };
    }

    make_test!(f32);
    make_test!(f64);

    make_test_signed!(f32);
    make_test_signed!(f64);
}
