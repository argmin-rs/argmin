use crate::faer_tests::test_helper::*;
use crate::ArgminDot;
use approx::assert_relative_eq;
use faer::mat::AsMatRef;
use faer::Col;
use paste::item;

macro_rules! make_test {
    ($t:ty) => {
        item! {
            #[test]
            fn [<test_col_col_ $t>]() {
                let a = col3_new(1 as $t, 2 as $t, 3 as $t);
                let b = col3_new(4 as $t, 5 as $t, 6 as $t);
                // all owned and reference type combinations
                let res1: $t = <_ as ArgminDot<_, _>>::dot(&a, &b);
                let res2: $t = <_ as ArgminDot<_, _>>::dot(&a.as_ref(), &b);
                let res3: $t = <_ as ArgminDot<_, _>>::dot(&a, &b.as_ref());
                let res4: $t = <_ as ArgminDot<_, _>>::dot(&a.as_ref(), &b.as_ref());
                assert_relative_eq!(res1 as f64, 32 as f64, epsilon = f64::EPSILON);
                assert_relative_eq!(res2 as f64, 32 as f64, epsilon = f64::EPSILON);
                assert_relative_eq!(res3 as f64, 32 as f64, epsilon = f64::EPSILON);
                assert_relative_eq!(res4 as f64, 32 as f64, epsilon = f64::EPSILON);
            }
        }

        item! {
            #[test]
            fn [<test_col_scalar_ $t>]() {
                let a = col3_new(1 as $t, 2 as $t, 3 as $t);
                let b = 2 as $t;
                let product1: Col<$t> =
                    <_ as ArgminDot<$t, _>>::dot(&a, &b);
                let product2: Col<$t> =
                    <_ as ArgminDot<$t, _>>::dot(&a.as_ref(), &b);
                let res = col3_new(2 as $t, 4 as $t, 6 as $t);
                assert_eq!(product1,product2);
                assert_eq!(product1.nrows(),3);
                assert_eq!(product1.ncols(),1);
                for i in 0..3 {
                    assert_relative_eq!(res[i] as f64, product1[i] as f64, epsilon = f64::EPSILON);
                }
            }
        }

        item! {
            #[test]
            fn [<test_scalar_col_ $t>]() {
                let a = col3_new(1 as $t, 2 as $t, 3 as $t);
                let b = 2 as $t;
                let product1: Col<$t> =
                    <$t as ArgminDot<_, _>>::dot(&b, &a);
                let product2: Col<$t> =
                    <$t as ArgminDot<_, _>>::dot(&b, &a.as_ref());
                assert_eq!(product1,product2);
                assert_eq!(product1.nrows(),3);
                assert_eq!(product1.ncols(),1);
                let res = col3_new(2 as $t, 4 as $t, 6 as $t);
                for i in 0..3 {
                    assert_relative_eq!(res[i] as f64, product1[i] as f64, epsilon = f64::EPSILON);
                }
            }
        }

        item! {
            #[test] // corresponds to test_mat_vec_2_ in super
            fn [<test_mat_col_ $t>]() {
                let a = matrix3_new(
                    1 as $t, 2 as $t, 3 as $t,
                    4 as $t, 5 as $t, 6 as $t,
                    7 as $t, 8 as $t, 9 as $t
                );
                let b = col3_new(1 as $t, 2 as $t, 3 as $t);
                let res = col3_new(14 as $t, 32 as $t, 50 as $t);
                let product1: Col<$t> =
                    <_ as ArgminDot<_, _>>::dot(&a, &b);
                let product2: Col<$t> =
                    <_ as ArgminDot<_, _>>::dot(&a.as_mat_ref(), &b);
                let product3: Col<$t> =
                    <_ as ArgminDot<_, _>>::dot(&a, &b.as_ref());
                let product4: Col<$t> =
                    <_ as ArgminDot<_, _>>::dot(&a.as_mat_ref(), &b.as_ref());
                for i in 0..3 {
                    assert_relative_eq!(res[i] as f64, product1[i] as f64, epsilon = f64::EPSILON);
                    assert_relative_eq!(res[i] as f64, product2[i] as f64, epsilon = f64::EPSILON);
                    assert_relative_eq!(res[i] as f64, product3[i] as f64, epsilon = f64::EPSILON);
                    assert_relative_eq!(res[i] as f64, product4[i] as f64, epsilon = f64::EPSILON);
                }
            }
        }
    };
}

make_test!(f32);
make_test!(f64);
