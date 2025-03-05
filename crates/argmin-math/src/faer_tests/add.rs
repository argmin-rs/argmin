use crate::faer_tests::test_helper::*;
use crate::ArgminAdd;
use approx::assert_relative_eq;
use faer::mat::AsMatRef;
use paste::item;

macro_rules! make_test {
    ($t:ty) => {
        item! {
            #[test]
            fn [<test_add_vec_scalar_ $t>]() {
                let a = vector3_new(1 as $t, 4 as $t, 8 as $t);
                let b = 34 as $t;
                let target = vector3_new(35 as $t, 38 as $t, 42 as $t);
                let res1 = <_ as ArgminAdd<$t, _>>::add(&a, &b);
                let res2 = <_ as ArgminAdd<$t, _>>::add(&a.as_mat_ref(), &b);
                assert_eq!(res1, res2);
                assert_eq!(res1.nrows(), 3);
                assert_eq!(res1.ncols(), 1);
                for i in 0..3 {
                    assert_relative_eq!(target[(i,0)] as f64, res1[(i,0)] as f64, epsilon = f64::EPSILON);
                }
            }
        }

        item! {
            #[test]
            fn [<test_add_scalar_vec_ $t>]() {
                let a = vector3_new(1 as $t, 4 as $t, 8 as $t);
                let b = 34 as $t;
                let target = vector3_new(35 as $t, 38 as $t, 42 as $t);
                let res1 = <_ as ArgminAdd<_, _>>::add(&b, &a);
                let res2 = <_ as ArgminAdd<_, _>>::add(&b, &a.as_mat_ref());
                assert_eq!(res1, res2);
                assert_eq!(res1.nrows(), 3);
                assert_eq!(res1.ncols(), 1);
                for i in 0..3 {
                    assert_relative_eq!(target[(i,0)] as f64, res1[(i,0)] as f64, epsilon = f64::EPSILON);
                }
            }
        }

        item! {
            #[test]
            fn [<test_add_vec_vec_ $t>]() {
                let a = vector3_new(1 as $t, 4 as $t, 8 as $t);
                let b = vector3_new(41 as $t, 38 as $t, 34 as $t);
                let target = vector3_new(42 as $t, 42 as $t, 42 as $t);
                let res = <_ as ArgminAdd<_, _>>::add(&a, &b);
                for i in 0..3 {
                    assert_relative_eq!(target[(i,0)] as f64, res[(i,0)] as f64, epsilon = f64::EPSILON);
                }
            }
        }

        item! {
            #[test]
            #[should_panic]
            fn [<test_add_vec_vec_panic_ $t>]() {
                let a = column_vector_from_vec(vec![1 as $t, 4 as $t]);
                let b = column_vector_from_vec(vec![41 as $t, 38 as $t, 34 as $t]);
                <_ as ArgminAdd<_,_>>::add(&a, &b);
            }
        }

        item! {
            #[test]
            #[should_panic]
            fn [<test_add_vec_vec_panic_2_ $t>]() {
                let a = column_vector_from_vec(vec![]);
                let b = column_vector_from_vec(vec![41 as $t, 38 as $t, 34 as $t]);
                <_ as ArgminAdd<_, _>>::add(&a, &b);
            }
        }

        item! {
            #[test]
            #[should_panic]
            fn [<test_add_vec_vec_panic_3_ $t>]() {
                let a = column_vector_from_vec(vec![41 as $t, 38 as $t, 34 as $t]);
                let b = column_vector_from_vec(vec![]);
                <_ as ArgminAdd<_, _>>::add(&a, &b);
            }
        }

        item! {
            #[test]
            fn [<test_add_mat_mat_ $t>]() {
                let a = matrix2x3_new(
                    1 as $t, 4 as $t, 8 as $t,
                    2 as $t, 5 as $t, 9 as $t
                );
                let b = matrix2x3_new(
                    41 as $t, 38 as $t, 34 as $t,
                    40 as $t, 37 as $t, 33 as $t
                );
                let target = matrix2x3_new(
                    42 as $t, 42 as $t, 42 as $t,
                    42 as $t, 42 as $t, 42 as $t
                );
                let res1 = <_ as ArgminAdd<_, _>>::add(&a, &b);
                let res2 = <_ as ArgminAdd<_, _>>::add(&a.as_mat_ref(), &b);
                let res3 = <_ as ArgminAdd<_, _>>::add(&a, &b.as_mat_ref());
                let res4 = <_ as ArgminAdd<_, _>>::add(&a.as_mat_ref(), &b.as_mat_ref());
                assert_eq!(res1, res2);
                assert_eq!(res1, res3);
                assert_eq!(res1, res4);
                assert_eq!(res1.nrows(), 2);
                assert_eq!(res1.ncols(), 3);
                for i in 0..3 {
                    for j in 0..2 {
                        assert_relative_eq!(target[(j, i)] as f64, res1[(j, i)] as f64, epsilon = f64::EPSILON);
                    }
                }
            }
        }

        item! {
            #[test]
            fn [<test_add_mat_scalar_ $t>]() {
                let a = matrix2x3_new(
                    1 as $t, 4 as $t, 8 as $t,
                    2 as $t, 5 as $t, 9 as $t
                );
                let b = 2 as $t;
                let target = matrix2x3_new(
                    3 as $t, 6 as $t, 10 as $t,
                    4 as $t, 7 as $t, 11 as $t
                );
                let res1 = <_ as ArgminAdd<$t, _>>::add(&a, &b);
                let res2 = <_ as ArgminAdd<$t, _>>::add(&a.as_mat_ref(), &b);
                assert_eq!(res1, res2);
                assert_eq!(res1.nrows(), 2);
                assert_eq!(res1.ncols(), 3);
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
            fn [<test_add_mat_mat_panic_2_ $t>]() {
                let a = faer::mat![
                    [1 as $t, 4 as $t, 8 as $t],
                    [2 as $t, 5 as $t, 9 as $t]
                ];
                let b = faer::mat![
                    [41 as $t, 38 as $t]
                ];
                <_ as ArgminAdd<_, _>>::add(&a, &b);
            }
        }

        item! {
            #[test]
            #[should_panic]
            fn [<test_add_mat_mat_panic_3_ $t>]() {
                let a = faer::mat![
                    [1 as $t, 4 as $t, 8 as $t],
                    [2 as $t, 5 as $t, 9 as $t]
                ];
                let b = faer::Mat::new();
                <_ as ArgminAdd<_, _>>::add(&a, &b);
            }
        }
    };
}

make_test!(f32);
make_test!(f64);
