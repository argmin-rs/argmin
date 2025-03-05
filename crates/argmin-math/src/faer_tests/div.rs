use crate::faer_tests::test_helper::*;
use crate::ArgminDiv;
use approx::assert_relative_eq;
use faer::mat::AsMatRef;
use faer::Mat;
use paste::item;

macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_div_vec_scalar_ $t>]() {
                    let a = vector3_new(4 as $t, 16 as $t, 8 as $t);
                    let b = 2 as $t;
                    let target = vector3_new(2 as $t, 8 as $t, 4 as $t);
                    let res1 = <_ as ArgminDiv<$t, _>>::div(&a, &b);
                    let res2 = <_ as ArgminDiv<$t, _>>::div(&a.as_mat_ref(), &b);
                    assert_eq!(res1,res2);
                    assert_eq!(res1.nrows(),3);
                    assert_eq!(res1.ncols(),1);
                    for i in 0..3 {
                        assert_relative_eq!(target[(i,0)] as f64, res1[(i,0)] as f64, epsilon = f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_div_scalar_vec_ $t>]() {
                    let a = vector3_new(2 as $t, 4 as $t, 8 as $t);
                    let b = 32 as $t;
                    let target = vector3_new(16 as $t, 8 as $t, 4 as $t);
                    let res1 = <$t as ArgminDiv<_, _>>::div(&b, &a);
                    let res2 = <$t as ArgminDiv<_, _>>::div(&b, &a);
                    assert_eq!(res1,res2);
                    assert_eq!(res1.nrows(),3);
                    assert_eq!(res1.ncols(),1);
                    for i in 0..3 {
                        assert_relative_eq!(target[(i,0)] as f64, res1[(i,0)] as f64, epsilon = f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_div_vec_vec_ $t>]() {
                    let a = vector3_new(4 as $t, 9 as $t, 8 as $t);
                    let b = vector3_new(2 as $t, 3 as $t, 4 as $t);
                    let target = vector3_new(2 as $t, 3 as $t, 2 as $t);
                    let res1 :Mat<$t> = <_ as ArgminDiv<_, _>>::div(&a, &b);
                    let res2 :Mat<$t> = <_ as ArgminDiv<_, _>>::div(&a, &b);
                    let res3 :Mat<$t> = <_ as ArgminDiv<_, _>>::div(&a, &b);
                    let res4 :Mat<$t> = <_ as ArgminDiv<_, _>>::div(&a, &b);
                    assert_eq!(res1,res2);
                    assert_eq!(res1,res3);
                    assert_eq!(res1,res4);
                    assert_eq!(res1.nrows(),3);
                    assert_eq!(res1.ncols(),1);
                    for i in 0..3 {
                        assert_relative_eq!(target[(i,0)] as f64, res1[(i,0)] as f64, epsilon = f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_div_vec_vec_panic_ $t>]() {
                    let a = column_vector_from_vec(vec![1 as $t, 4 as $t]);
                    let b = column_vector_from_vec(vec![41 as $t, 38 as $t, 34 as $t]);
                    <_ as ArgminDiv<_, _>>::div(&a, &b);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_div_vec_vec_panic_2_ $t>]() {
                    let a = column_vector_from_vec(vec![]); let b = column_vector_from_vec(vec![41 as $t, 38 as $t, 34 as $t]); <_ as ArgminDiv<_, _>>::div(&a, &b);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_div_vec_vec_panic_3_ $t>]() {
                    let a = column_vector_from_vec(vec![41 as $t, 38 as $t, 34 as $t]);
                    let b = column_vector_from_vec(vec![]);
                    <_ as ArgminDiv<_,_>>::div(&a, &b);
                }
            }

            item! {
                #[test]
                fn [<test_div_mat_mat_ $t>]() {
                    let a = matrix2x3_new(
                        4 as $t, 12 as $t, 8 as $t,
                        9 as $t, 20 as $t, 45 as $t
                    );
                    let b = matrix2x3_new(
                        2 as $t, 3 as $t, 4 as $t,
                        3 as $t, 4 as $t, 5 as $t
                    );
                    let target = matrix2x3_new(
                        2 as $t, 4 as $t, 2 as $t,
                        3 as $t, 5 as $t, 9 as $t
                    );
                    let res1 = <_ as ArgminDiv<_, _>>::div(&a, &b);
                    let res2 = <_ as ArgminDiv<_, _>>::div(&a, &b);
                    let res3 = <_ as ArgminDiv<_, _>>::div(&a, &b);
                    let res4 = <_ as ArgminDiv<_, _>>::div(&a, &b);
                    assert_eq!(res1,res2);
                    assert_eq!(res1,res3);
                    assert_eq!(res1,res4);
                    assert_eq!(res1.nrows(),2);
                    assert_eq!(res1.ncols(),3);
                    for i in 0..3 {
                        for j in 0..2 {
                            assert_relative_eq!(target[(j, i)] as f64, res1[(j, i)] as f64, epsilon = f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_div_mat_mat_panic_2_ $t>]() {
                    let a = matrix2x3_new(
                        1 as $t, 4 as $t, 8 as $t,
                        2 as $t, 5 as $t, 9 as $t
                    );
                    let b = faer::mat![
                        [41 as $t, 38 as $t]
                    ];
                    <_ as ArgminDiv<_, _>>::div(&a, &b);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_div_mat_mat_panic_3_ $t>]() {
                    let a = matrix2x3_new(
                        1 as $t, 4 as $t, 8 as $t,
                        2 as $t, 5 as $t, 9 as $t
                    );
                    let b = Mat::new();
                    <_ as ArgminDiv<_, _>>::div(&a, &b);
                }
            }
        };
    }

make_test!(f32);
make_test!(f64);
