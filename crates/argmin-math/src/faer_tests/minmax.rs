use crate::faer_tests::test_helper::*;
use crate::ArgminMinMax;
use approx::assert_relative_eq;
use paste::item;

macro_rules! make_test {
    ($t:ty) => {
        item! {
            #[test]
            fn [<test_minmax_vec_vec_ $t>]() {
                let a = vector3_new(1 as $t, 4 as $t, 8 as $t);
                let b = vector3_new(2 as $t, 3 as $t, 4 as $t);
                let target_max = vector3_new(2 as $t, 4 as $t, 8 as $t);
                let target_min = vector3_new(1 as $t, 3 as $t, 4 as $t);
                let res_max = <_ as ArgminMinMax>::max(&a, &b);
                let res_min = <_ as ArgminMinMax>::min(&a, &b);
                for i in 0..3 {
                    assert_relative_eq!(target_max[(i,0)] as f64, res_max[(i,0)] as f64, epsilon = f64::EPSILON);
                    assert_relative_eq!(target_min[(i,0)] as f64, res_min[(i,0)] as f64, epsilon = f64::EPSILON);
                }
            }
        }

        item! {
            #[test]
            fn [<test_minmax_mat_mat_ $t>]() {
                let a = matrix2x3_new(
                    1 as $t, 4 as $t, 8 as $t,
                    2 as $t, 5 as $t, 9 as $t
                );
                let b = matrix2x3_new(
                    2 as $t, 3 as $t, 4 as $t,
                    3 as $t, 4 as $t, 5 as $t
                );
                let target_max = matrix2x3_new(
                    2 as $t, 4 as $t, 8 as $t,
                    3 as $t, 5 as $t, 9 as $t
                );
                let target_min = matrix2x3_new(
                    1 as $t, 3 as $t, 4 as $t,
                    2 as $t, 4 as $t, 5 as $t
                );
                let res_max = <_ as ArgminMinMax>::max(&a, &b);
                let res_min = <_ as ArgminMinMax>::min(&a, &b);
                for i in 0..3 {
                    for j in 0..2 {
                        assert_relative_eq!(target_max[(j, i)] as f64, res_max[(j, i)] as f64, epsilon = f64::EPSILON);
                        assert_relative_eq!(target_min[(j, i)] as f64, res_min[(j, i)] as f64, epsilon = f64::EPSILON);
                    }
                }
            }
        }
    };
}

make_test!(f32);
make_test!(f64);
