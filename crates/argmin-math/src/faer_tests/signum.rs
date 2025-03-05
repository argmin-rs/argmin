use crate::faer_tests::test_helper::*;
use crate::ArgminSignum;
use approx::assert_relative_eq;
use faer::mat;
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
