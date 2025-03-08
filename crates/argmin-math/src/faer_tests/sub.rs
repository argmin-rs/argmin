use crate::faer_tests::test_helper::*;
use crate::ArgminSub;
use approx::assert_relative_eq;
use faer::mat;
use faer::mat::AsMatRef;
use faer::Mat;
use paste::item;

macro_rules! make_test {
    ($t:ty) => {
        item! {
            #[test]
            fn [<test_sub_vec_scalar_ $t>]() {
                let a: Mat<$t> = column_vector_from_vec(vec![36 as $t, 39 as $t, 43 as $t]);
                let b: $t = 1 as $t;
                let target: Mat<$t> = column_vector_from_vec(vec![35 as $t, 38 as $t, 42 as $t]);
                // make sure we get the same answer regardless whether we
                // use owned matrices or matrix references.
                let res = <_ as ArgminSub<_,_>>::sub(&a, &b);
                let res2 = <_ as ArgminSub<_,_>>::sub(&a.as_mat_ref(), &b);
                assert_eq!(res.nrows(),3);
                assert_eq!(res.ncols(),1);
                assert_eq!(res,res2);
                for i in 0..3 {
                    assert_relative_eq!(target[(i,0)] as f64, res[(i,0)] as f64, epsilon = f64::EPSILON);
                }
            }
        }

        item! {
            #[test]
            fn [<test_sub_scalar_vec_ $t>]() {
                let a = column_vector_from_vec(vec![1 as $t, 4 as $t, 8 as $t]);
                let b = 34 as $t;
                let target = column_vector_from_vec(vec![33 as $t, 30 as $t, 26 as $t]);
                let res = <$t as ArgminSub<_,_>>::sub(&b, &a);
                let res2 = <$t as ArgminSub<_,_>>::sub(&b, &a.as_mat_ref());
                assert_eq!(res.nrows(),3);
                assert_eq!(res.ncols(),1);
                assert_eq!(res,res2);
                for i in 0..3 {
                    assert_relative_eq!(target[(i,0)] as f64, res[(i,0)] as f64, epsilon = f64::EPSILON);
                }
            }
        }

        item! {
            #[test]
            fn [<test_sub_vec_vec_ $t>]() {
                let a: Mat<$t> = column_vector_from_vec(vec![41 as $t, 38 as $t, 34 as $t]);
                let b: Mat<$t> = column_vector_from_vec(vec![1 as $t, 4 as $t, 8 as $t]);
                let target: Mat<$t> = column_vector_from_vec(vec![40 as $t, 34 as $t, 26 as $t]);
                // all combinations of references and owned matrices
                let res = <_ as ArgminSub<_,_>>::sub(&a, &b);
                let res2 = <_ as ArgminSub<_,_>>::sub(&a.as_mat_ref(), &b);
                let res3 = <_ as ArgminSub<_,_>>::sub(&a, &b.as_mat_ref());
                let res4 = <_ as ArgminSub<_,_>>::sub(&a.as_mat_ref(), &b.as_mat_ref());
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
            fn [<test_sub_mat_mat_ $t>]() {
                let a = mat![
                    [43 as $t, 46 as $t, 50 as $t],
                    [44 as $t, 47 as $t, 51 as $t]
                ];
                let b = mat![
                    [1 as $t, 4 as $t, 8 as $t],
                   [ 2 as $t, 5 as $t, 9 as $t]
                ];
                let target = mat![
                    [42 as $t, 42 as $t, 42 as $t],
                    [42 as $t, 42 as $t, 42 as $t]
                ];
                // all possible combinations of owned matrices and references
                let res = <_ as ArgminSub<_,_>>::sub(&a, &b);
                let res2 = <_ as ArgminSub<_,_>>::sub(&a.as_mat_ref(), &b);
                let res3 = <_ as ArgminSub<_,_>>::sub(&a, &b.as_mat_ref());
                let res4 = <_ as ArgminSub<_,_>>::sub(&a.as_mat_ref(), &b.as_mat_ref());
                assert_eq!(res.nrows(),2);
                assert_eq!(res.ncols(),3);
                assert_eq!(res,res2);
                assert_eq!(res,res3);
                assert_eq!(res,res4);
                for i in 0..3 {
                    for j in 0..2 {
                        assert_relative_eq!(target[(j, i)] as f64, res[(j, i)] as f64, epsilon = f64::EPSILON);
                    }
                }
            }
        }

        item! {
            #[test]
            fn [<test_sub_mat_scalar_ $t>]() {
                let a = mat![
                    [43 as $t, 46 as $t, 50 as $t],
                    [44 as $t, 47 as $t, 51 as $t]
                ];
                let b = 2 as $t;
                let target = mat![
                    [41 as $t, 44 as $t, 48 as $t],
                    [42 as $t, 45 as $t, 49 as $t]
                ];
                // combinations of references and owned matrices
                let res = <_ as ArgminSub<_,_>>::sub(&a, &b);
                let res2 = <_ as ArgminSub<_,_>>::sub(&a.as_mat_ref(), &b);
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
