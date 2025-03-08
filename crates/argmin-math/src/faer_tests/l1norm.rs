use crate::faer_tests::test_helper::*;
use crate::ArgminL1Norm;
use approx::assert_relative_eq;
use paste::item;

macro_rules! make_test {
    ($t:ty) => {
        item! {
            #[test]
            fn [<test_l1norm_ $t>]() {
                let a = vector2_new(4 as $t, 3 as $t);
                let res = <_ as ArgminL1Norm<$t>>::l1_norm(&a);
                let target = 7 as $t;
                assert_relative_eq!(target as $t, res as $t, epsilon = $t::EPSILON);
            }
        }
    };
}

macro_rules! make_test_signed {
    ($t:ty) => {
        item! {
            #[test]
            fn [<test_l1norm_signed_ $t>]() {
                let a = vector2_new(-4 as $t, -3 as $t);
                let res = <_ as ArgminL1Norm<$t>>::l1_norm(&a);
                let target = 7 as $t;
                assert_relative_eq!(target as $t, res as $t, epsilon = $t::EPSILON);
            }
        }
    };
}

make_test!(f32);
make_test!(f64);

make_test_signed!(f32);
make_test_signed!(f64);
