use crate::faer_tests::test_helper::*;
use crate::ArgminTranspose;
use approx::assert_relative_eq;
use faer::mat;
use faer::mat::AsMatRef;
use faer::{Mat, MatRef};
use paste::item;

macro_rules! make_test {
    ($t:ty) => {
        item! {
            #[test]
            fn [<test_transpose_ $t>]() {
                // we make sure that transposition works for both references
                // and owned matrices
                let a : Mat<$t> = column_vector_from_vec(vec![1 as $t, 4 as $t]);
                let a2 = a.clone();
                let a3 = a.clone();
                assert_eq!(a.nrows(),2);
                assert_eq!(a.ncols(),1);
                //@note(geo-ant) test all possible implementations
                let res =  <_ as ArgminTranspose<Mat<_>>>::t(a);
                let res2 = <_ as ArgminTranspose<Mat<_>>>::t(a2.as_mat_ref());
                let res3 =  <_ as ArgminTranspose<MatRef<_>>>::t(&a3);
                let res4 =  <_ as ArgminTranspose<MatRef<_>>>::t(a3.as_mat_ref());
                assert_eq!(res.nrows(),1);
                assert_eq!(res.ncols(),2);
                assert_eq!(res,res2);
                assert_eq!(res,res3);
                assert_eq!(res,res4);
                assert_relative_eq!(res[(0,0)] as f64, 1. as f64, epsilon = f64::EPSILON);
            }
        }

        item! {
            #[test]
            fn [<test_transpose_2d_1_ $t>]() {
                let a :Mat<$t> = mat![
                    [1 as $t, 4 as $t],
                    [8 as $t, 7 as $t]
                ];
                let a2 = a.clone();
                let a3 = a.clone();
                let target = mat![
                    [1 as $t, 8 as $t],
                    [4 as $t, 7 as $t]
                ];
                let res =  <_ as ArgminTranspose<Mat<_>>>::t(a);
                let res2 = <_ as ArgminTranspose<Mat<_>>>::t(a2.as_mat_ref());
                let res3 =  <_ as ArgminTranspose<MatRef<_>>>::t(&a3);
                let res4 =  <_ as ArgminTranspose<MatRef<_>>>::t(a3.as_mat_ref());
                assert_eq!(res.nrows(),2);
                assert_eq!(res.ncols(),2);
                assert_eq!(res,res2);
                assert_eq!(res,res3);
                assert_eq!(res,res4);
                for i in 0..2 {
                    for j in 0..2 {
                        assert_relative_eq!(target[(i, j)] as f64, res[(i, j)] as f64, epsilon = f64::EPSILON);
                    }
                }
            }
        }

        item! {
            #[test]
            fn [<test_transpose_2d_2_ $t>]() {
                let a : Mat<$t>= mat![
                    [1 as $t, 4 as $t],
                    [8 as $t, 7 as $t],
                    [3 as $t, 6 as $t]
                ];
                let a2 = a.clone();
                let a3 = a.clone();
                let target = mat![
                    [1 as $t, 8 as $t, 3 as $t],
                    [4 as $t, 7 as $t, 6 as $t]
                ];
                let res =  <_ as ArgminTranspose<Mat<_>>>::t(a);
                let res2 = <_ as ArgminTranspose<Mat<_>>>::t(a2.as_mat_ref());
                let res3 =  <_ as ArgminTranspose<MatRef<_>>>::t(&a3);
                let res4 =  <_ as ArgminTranspose<MatRef<_>>>::t(a3.as_mat_ref());
                assert_eq!(res.nrows(),target.nrows());
                assert_eq!(res.ncols(),target.ncols());
                assert_eq!(res,res2);
                assert_eq!(res,res3);
                assert_eq!(res,res4);
                for i in 0..2 {
                    for j in 0..3 {
                        assert_relative_eq!(target[(i, j)] as f64, res[(i, j)] as f64, epsilon = f64::EPSILON);
                    }
                }
            }
        }
    };
}

make_test!(f32);
make_test!(f64);
