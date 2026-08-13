use crate::faer_tests::test_helper::*;
use crate::ArgminSub;
use approx::assert_relative_eq;
use faer::Col;
use paste::item;

macro_rules! make_test {
    ($t:ty) => {
        item! {
            #[test]
            fn [<test_sub_col_scalar_ $t>]() {
                let a: Col<$t> = col_from_vec(vec![36 as $t, 39 as $t, 43 as $t]);
                let b: $t = 1 as $t;
                let target: Col<$t> = col_from_vec(vec![35 as $t, 38 as $t, 42 as $t]);
                // make sure we get the same answer regardless whether we
                // use owned matrices or matrix references.
                let res = <_ as ArgminSub<_,_>>::sub(&a, &b);
                let res2 = <_ as ArgminSub<_,_>>::sub(&a.as_ref(), &b);
                assert_eq!(res.nrows(),3);
                assert_eq!(res,res2);
                for i in 0..3 {
                    assert_relative_eq!(target[i] as f64, res[i] as f64, epsilon = f64::EPSILON);
                }
            }
        }

        item! {
            #[test]
            fn [<test_sub_scalar_col_ $t>]() {
                let a = 34 as $t;
                let b = col_from_vec(vec![1 as $t, 4 as $t, 8 as $t]);
                let target = col_from_vec(vec![33 as $t, 30 as $t, 26 as $t]);
                let res = <$t as ArgminSub<_,_>>::sub(&a, &b);
                let res2 = <$t as ArgminSub<_,_>>::sub(&a, &b.as_ref());
                assert_eq!(res.nrows(),3);
                assert_eq!(res,res2);
                for i in 0..3 {
                    assert_relative_eq!(target[i] as f64, res[i] as f64, epsilon = f64::EPSILON);
                }
            }
        }

        item! {
            #[test]
            fn [<test_sub_col_col_ $t>]() {
                let a: Col<$t> = col_from_vec(vec![41 as $t, 38 as $t, 34 as $t]);
                let b: Col<$t> = col_from_vec(vec![1 as $t, 4 as $t, 8 as $t]);
                let target: Col<$t> = col_from_vec(vec![40 as $t, 34 as $t, 26 as $t]);
                // all combinations of references and owned matrices
                let res = <_ as ArgminSub<_,_>>::sub(&a, &b);
                let res2 = <_ as ArgminSub<_,_>>::sub(&a.as_ref(), &b);
                let res3 = <_ as ArgminSub<_,_>>::sub(&a, &b.as_ref());
                let res4 = <_ as ArgminSub<_,_>>::sub(&a.as_ref(), &b.as_ref());
                assert_eq!(res.nrows(),3);
                assert_eq!(res,res2);
                assert_eq!(res,res3);
                assert_eq!(res,res4);
                for i in 0..3 {
                    assert_relative_eq!(target[i] as f64, res[i] as f64, epsilon = f64::EPSILON);
                }
            }
        }
    };
}

make_test!(f32);
make_test!(f64);
