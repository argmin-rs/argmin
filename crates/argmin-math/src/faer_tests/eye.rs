use crate::faer_tests::test_helper::*;
use crate::ArgminEye;
use approx::assert_relative_eq;
use faer::Mat;
use paste::item;

macro_rules! make_test {
    ($t:ty) => {
        item! {
            #[test]
            fn [<test_eye_ $t>]() {
                let e: Mat<$t> = <_ as ArgminEye>::eye(3);
                let res = matrix3_new(
                    1 as $t, 0 as $t, 0 as $t,
                    0 as $t, 1 as $t, 0 as $t,
                    0 as $t, 0 as $t, 1 as $t
                );
                for i in 0..3 {
                    for j in 0..3 {
                        assert_relative_eq!(res[(i, j)] as f64, e[(i, j)] as f64, epsilon = f64::EPSILON);
                    }
                }
            }
        }

        item! {
            #[test]
            fn [<test_eye_like_ $t>]() {
                let a = matrix3_new(
                    0 as $t, 2 as $t, 6 as $t,
                    3 as $t, 2 as $t, 7 as $t,
                    9 as $t, 8 as $t, 1 as $t
                );
                let e: Mat<$t> = a.eye_like();
                let res = matrix3_new(
                    1 as $t, 0 as $t, 0 as $t,
                    0 as $t, 1 as $t, 0 as $t,
                    0 as $t, 0 as $t, 1 as $t
                );
                for i in 0..3 {
                    for j in 0..3 {
                        assert_relative_eq!(res[(i, j)] as f64, e[(i, j)] as f64, epsilon = f64::EPSILON);
                    }
                }
            }
        }
    };
}

make_test!(f32);
make_test!(f64);
