use crate::faer_tests::test_helper::*;
use crate::ArgminMul;
use approx::assert_relative_eq;
use faer::mat::AsMatRef;
use paste::item;

macro_rules! make_test {
    ($t:ty) => {
        item! {
            #[test]
            fn [<test_mul_vec_scalar_ $t>]() {
                let a = vector3_new(1 as $t, 4 as $t, 8 as $t);
                let b = 2 as $t;
                let target = vector3_new(2 as $t, 8 as $t, 16 as $t);
                let res = <_ as ArgminMul<_, _>>::mul(&a, &b);
                let res2 = <_ as ArgminMul<_, _>>::mul(&a.as_mat_ref(), &b);
                assert_eq!(res,res2);
                assert_eq!(res.nrows(),3);
                assert_eq!(res.ncols(),1);
                for i in 0..3 {
                    assert_relative_eq!(target[(i,0)] as f64, res[(i,0)] as f64, epsilon = f64::EPSILON);
                }
            }
        }

        item! {
            #[test]
            fn [<test_mul_scalar_vec_ $t>]() {
                let a = vector3_new(1 as $t, 4 as $t, 8 as $t);
                let b = 2 as $t;
                let target = vector3_new(2 as $t, 8 as $t, 16 as $t);
                let res = <_ as ArgminMul<_,_>>::mul(&b, &a);
                let res2 = <_ as ArgminMul<_, _>>::mul(&b, &a.as_mat_ref());
                assert_eq!(res,res2);
                assert_eq!(res.nrows(),3);
                assert_eq!(res.ncols(),1);
                for i in 0..3 {
                    assert_relative_eq!(target[(i,0)] as f64, res[(i,0)] as f64, epsilon = f64::EPSILON);
                }
            }
        }

        item! {
            #[test]
            fn [<test_mul_vec_vec_ $t>]() {
                let a = vector3_new(1 as $t, 4 as $t, 8 as $t);
                let b = vector3_new(2 as $t, 3 as $t, 4 as $t);
                let target = vector3_new(2 as $t, 12 as $t, 32 as $t);
                let res = <_ as ArgminMul<_,_>>::mul(&a, &b);
                let res2 = <_ as ArgminMul<_,_>>::mul(&a.as_mat_ref(), &b);
                let res3 = <_ as ArgminMul<_,_>>::mul(&a, &b.as_mat_ref());
                let res4 = <_ as ArgminMul<_,_>>::mul(&a.as_mat_ref(), &b.as_mat_ref());
                assert_eq!(res.nrows(),3);
                assert_eq!(res.ncols(),1);
                assert_eq!(res,res2);
                assert_eq!(res,res3);
                assert_eq!(res,res4);
                for i in 0..3 {
                    assert_relative_eq!(target[(i,0)] as f64, res[(i,0)] as f64, epsilon = f64::EPSILON);
                }
            }
        }

        item! {
            #[test]
            fn [<test_mul_mat_mat_ $t>]() {
                let a = matrix2x3_new(
                    1 as $t, 4 as $t, 8 as $t,
                    2 as $t, 5 as $t, 9 as $t
                );
                let b = matrix2x3_new(
                    2 as $t, 3 as $t, 4 as $t,
                    3 as $t, 4 as $t, 5 as $t
                );
                let target = matrix2x3_new(
                    2 as $t, 12 as $t, 32 as $t,
                    6 as $t, 20 as $t, 45 as $t
                );
                let res = <_ as ArgminMul<_,_>>::mul(&a, &b);
                let res2 = <_ as ArgminMul<_,_>>::mul(&a.as_mat_ref(), &b);
                let res3 = <_ as ArgminMul<_,_>>::mul(&a, &b.as_mat_ref());
                let res4 = <_ as ArgminMul<_,_>>::mul(&a.as_mat_ref(), &b.as_mat_ref());
                assert_eq!(res.nrows(),2);
                assert_eq!(res.ncols(),3);
                assert_eq!(res,res2);
                assert_eq!(res,res3);
                assert_eq!(res,res4);
                let res = <_ as ArgminMul<_,_>>::mul(&a, &b);
                for i in 0..3 {
                    for j in 0..2 {
                        assert_relative_eq!(target[(j, i)] as f64, res[(j, i)] as f64, epsilon = f64::EPSILON);
                    }
                }
            }
        }

        item! {
            #[test]
            fn [<test_mul_scalar_mat_1_ $t>]() {
                let a = matrix2x3_new(
                    1 as $t, 4 as $t, 8 as $t,
                    2 as $t, 5 as $t, 9 as $t
                );
                let b = 2 as $t;
                let target = matrix2x3_new(
                    2 as $t, 8 as $t, 16 as $t,
                    4 as $t, 10 as $t, 18 as $t
                );
                let res = <_ as ArgminMul<_,_>>::mul(&a, &b);
                let res2 = <_ as ArgminMul<_,_>>::mul(&a.as_mat_ref(), &b);
                assert_eq!(res.nrows(),2);
                assert_eq!(res.ncols(),3);
                assert_eq!(res,res2);
                for i in 0..3 {
                    for j in 0..2 {
                        assert_relative_eq!(target[(j, i)] as f64, res[(j, i)] as f64, epsilon = f64::EPSILON);
                    }
                }
            }
        }

        item! {
            #[test]
            fn [<test_mul_scalar_mat_2_ $t>]() {
                let b = matrix2x3_new(
                    1 as $t, 4 as $t, 8 as $t,
                    2 as $t, 5 as $t, 9 as $t
                );
                let a = 2 as $t;
                let target = matrix2x3_new(
                    2 as $t, 8 as $t, 16 as $t,
                    4 as $t, 10 as $t, 18 as $t
                );
                let res = <$t as ArgminMul<_,_>>::mul(&a, &b);
                let res2 = <$t as ArgminMul<_,_>>::mul(&a, &b.as_mat_ref());
                assert_eq!(res.nrows(),2);
                assert_eq!(res.ncols(),3);
                assert_eq!(res,res2);
                for i in 0..3 {
                    for j in 0..2 {
                        assert_relative_eq!(target[(j, i)] as f64, res[(j, i)] as f64, epsilon = f64::EPSILON);
                    }
                }
            }
        }
    };
}

make_test!(f32);
make_test!(f64);
