use crate::faer_tests::test_helper::*;
use crate::ArgminMul;
use approx::assert_relative_eq;
use paste::item;

macro_rules! make_test {
    ($t:ty) => {
        item! {
            #[test]
            fn [<test_mul_col_scalar_ $t>]() {
                let a = col3_new(1 as $t, 4 as $t, 8 as $t);
                let b = 2 as $t;
                let target = col3_new(2 as $t, 8 as $t, 16 as $t);
                let res = <_ as ArgminMul<_, _>>::mul(&a, &b);
                let res2 = <_ as ArgminMul<_, _>>::mul(&a.as_ref(), &b);
                assert_eq!(res,res2);
                assert_eq!(res.nrows(),3);
                assert_eq!(res.ncols(),1);
                for i in 0..3 {
                    assert_relative_eq!(target[i] as f64, res[i] as f64, epsilon = f64::EPSILON);
                }
            }
        }

        item! {
            #[test]
            fn [<test_mul_scalar_col_ $t>]() {
                let a = col3_new(1 as $t, 4 as $t, 8 as $t);
                let b = 2 as $t;
                let target = col3_new(2 as $t, 8 as $t, 16 as $t);
                let res = <_ as ArgminMul<_,_>>::mul(&b, &a);
                let res2 = <_ as ArgminMul<_, _>>::mul(&b, &a.as_ref());
                assert_eq!(res,res2);
                assert_eq!(res.nrows(),3);
                assert_eq!(res.ncols(),1);
                for i in 0..3 {
                    assert_relative_eq!(target[i] as f64, res[i] as f64, epsilon = f64::EPSILON);
                }
            }
        }

        item! {
            #[test]
            fn [<test_mul_col_col_ $t>]() {
                let a = col3_new(1 as $t, 4 as $t, 8 as $t);
                let b = col3_new(2 as $t, 3 as $t, 4 as $t);
                let target = col3_new(2 as $t, 12 as $t, 32 as $t);
                let res = <_ as ArgminMul<_,_>>::mul(&a, &b);
                let res2 = <_ as ArgminMul<_,_>>::mul(&a.as_ref(), &b);
                let res3 = <_ as ArgminMul<_,_>>::mul(&a, &b.as_ref());
                let res4 = <_ as ArgminMul<_,_>>::mul(&a.as_ref(), &b.as_ref());
                assert_eq!(res.nrows(),3);
                assert_eq!(res.ncols(),1);
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
