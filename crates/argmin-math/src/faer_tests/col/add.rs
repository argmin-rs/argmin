use crate::faer_tests::test_helper::*;
use crate::ArgminAdd;
use approx::assert_relative_eq;
use paste::item;

macro_rules! make_test {
    ($t:ty) => {
        item! {
            #[test]
            fn [<test_add_col_scalar_ $t>]() {
                let a = col3_new(1 as $t, 4 as $t, 8 as $t);
                let b = 34 as $t;
                let target = col3_new(35 as $t, 38 as $t, 42 as $t);
                let res1 = <_ as ArgminAdd<$t, _>>::add(&a, &b);
                let res2 = <_ as ArgminAdd<$t, _>>::add(&a.as_ref(), &b);
                assert_eq!(res1, res2);
                assert_eq!(res1.nrows(), 3);
                for i in 0..3 {
                    assert_relative_eq!(target[i] as f64, res1[i] as f64, epsilon = f64::EPSILON);
                }
            }
        }

        item! {
            #[test]
            fn [<test_add_scalar_col_ $t>]() {
                let a = col3_new(1 as $t, 4 as $t, 8 as $t);
                let b = 34 as $t;
                let target = col3_new(35 as $t, 38 as $t, 42 as $t);
                let res1 = <_ as ArgminAdd<_, _>>::add(&b, &a);
                let res2 = <_ as ArgminAdd<_, _>>::add(&b, &a.as_ref());
                assert_eq!(res1, res2);
                assert_eq!(res1.nrows(), 3);
                for i in 0..3 {
                    assert_relative_eq!(target[i] as f64, res1[i] as f64, epsilon = f64::EPSILON);
                }
            }
        }

        item! {
            #[test]
            fn [<test_add_col_col_ $t>]() {
                let a = col3_new(1 as $t, 4 as $t, 8 as $t);
                let b = col3_new(41 as $t, 38 as $t, 34 as $t);
                let target = col3_new(42 as $t, 42 as $t, 42 as $t);
                let res = <_ as ArgminAdd<_, _>>::add(&a, &b);
                for i in 0..3 {
                    assert_relative_eq!(target[i] as f64, res[i] as f64, epsilon = f64::EPSILON);
                }
            }
        }

        item! {
            #[test]
            #[should_panic]
            fn [<test_add_col_col_panic_ $t>]() {
                let a = col_from_vec(vec![1 as $t, 4 as $t]);
                let b = col_from_vec(vec![41 as $t, 38 as $t, 34 as $t]);
                <_ as ArgminAdd<_,_>>::add(&a, &b);
            }
        }

        item! {
            #[test]
            #[should_panic]
            fn [<test_add_col_col_panic_2_ $t>]() {
                let a = col_from_vec(vec![]);
                let b = col_from_vec(vec![41 as $t, 38 as $t, 34 as $t]);
                <_ as ArgminAdd<_, _>>::add(&a, &b);
            }
        }

        item! {
            #[test]
            #[should_panic]
            fn [<test_col_vec_col_panic_3_ $t>]() {
                let a = col_from_vec(vec![41 as $t, 38 as $t, 34 as $t]);
                let b = col_from_vec(vec![]);
                <_ as ArgminAdd<_, _>>::add(&a, &b);
            }
        }
    };
}

make_test!(f32);
make_test!(f64);
