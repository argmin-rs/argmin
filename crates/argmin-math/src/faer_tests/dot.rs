use crate::faer_tests::test_helper::*;
use crate::ArgminDot;
use approx::assert_relative_eq;
use faer::mat::AsMatRef;
use faer::Mat;
use paste::item;

macro_rules! make_test {
    ($t:ty) => {
        item! {
            #[test]
            fn [<test_vec_vec_ $t>]() {
                let a = vector3_new(1 as $t, 2 as $t, 3 as $t);
                let b = vector3_new(4 as $t, 5 as $t, 6 as $t);
                // all owned and reference type combinations
                let res1: $t = <_ as ArgminDot<_, _>>::dot(&a, &b);
                let res2: $t = <_ as ArgminDot<_, _>>::dot(&a.as_mat_ref(), &b);
                let res3: $t = <_ as ArgminDot<_, _>>::dot(&a, &b.as_mat_ref());
                let res4: $t = <_ as ArgminDot<_, _>>::dot(&a.as_mat_ref(), &b.as_mat_ref());
                assert_relative_eq!(res1 as f64, 32 as f64, epsilon = f64::EPSILON);
                assert_relative_eq!(res2 as f64, 32 as f64, epsilon = f64::EPSILON);
                assert_relative_eq!(res3 as f64, 32 as f64, epsilon = f64::EPSILON);
                assert_relative_eq!(res4 as f64, 32 as f64, epsilon = f64::EPSILON);
            }
        }

        item! {
            #[test]
            fn [<test_vec_scalar_ $t>]() {
                let a = vector3_new(1 as $t, 2 as $t, 3 as $t);
                let b = 2 as $t;
                let product1: Mat<$t> =
                    <_ as ArgminDot<$t, _>>::dot(&a, &b);
                let product2: Mat<$t> =
                    <_ as ArgminDot<$t, _>>::dot(&a.as_mat_ref(), &b);
                let res = vector3_new(2 as $t, 4 as $t, 6 as $t);
                assert_eq!(product1,product2);
                assert_eq!(product1.nrows(),3);
                assert_eq!(product1.ncols(),1);
                for i in 0..3 {
                    assert_relative_eq!(res[(i,0)] as f64, product1[(i,0)] as f64, epsilon = f64::EPSILON);
                }
            }
        }

        item! {
            #[test]
            fn [<test_scalar_vec_ $t>]() {
                let a = vector3_new(1 as $t, 2 as $t, 3 as $t);
                let b = 2 as $t;
                let product1: Mat<$t> =
                    <$t as ArgminDot<_, _>>::dot(&b, &a);
                let product2: Mat<$t> =
                    <$t as ArgminDot<_, _>>::dot(&b, &a.as_mat_ref());
                assert_eq!(product1,product2);
                assert_eq!(product1.nrows(),3);
                assert_eq!(product1.ncols(),1);
                let res = vector3_new(2 as $t, 4 as $t, 6 as $t);
                for i in 0..3 {
                    assert_relative_eq!(res[(i,0)] as f64, product1[(i,0)] as f64, epsilon = f64::EPSILON);
                }
            }
        }

        item! {
            #[test]
            fn [<test_mat_vec_ $t>]() {
                let a = vector3_new(1 as $t, 2 as $t, 3 as $t);
                let b = row_vector3_new(4 as $t, 5 as $t, 6 as $t);
                let res = matrix3_new(
                    4 as $t, 5 as $t, 6 as $t,
                    8 as $t, 10 as $t, 12 as $t,
                    12 as $t, 15 as $t, 18 as $t
                );
                let product1: Mat<$t> =
                    <_ as ArgminDot<_, _>>::dot(&a, &b);
                let product2: Mat<$t> =
                    <_ as ArgminDot<_, _>>::dot(&a.as_mat_ref(), &b);
                let product3: Mat<$t> =
                    <_ as ArgminDot<_, _>>::dot(&a, &b.as_mat_ref());
                let product4: Mat<$t> =
                    <_ as ArgminDot<_, _>>::dot(&a.as_mat_ref(), &b.as_mat_ref());
                for i in 0..3 {
                    for j in 0..3 {
                        assert_relative_eq!(res[(i, j)] as f64, product1[(i, j)] as f64, epsilon = f64::EPSILON);
                        assert_relative_eq!(res[(i, j)] as f64, product2[(i, j)] as f64, epsilon = f64::EPSILON);
                        assert_relative_eq!(res[(i, j)] as f64, product3[(i, j)] as f64, epsilon = f64::EPSILON);
                        assert_relative_eq!(res[(i, j)] as f64, product4[(i, j)] as f64, epsilon = f64::EPSILON);
                    }
                }
            }
        }

        item! {
            #[test]
            fn [<test_mat_vec_2_ $t>]() {
                let a = matrix3_new(
                    1 as $t, 2 as $t, 3 as $t,
                    4 as $t, 5 as $t, 6 as $t,
                    7 as $t, 8 as $t, 9 as $t
                );
                let b = vector3_new(1 as $t, 2 as $t, 3 as $t);
                let res = vector3_new(14 as $t, 32 as $t, 50 as $t);
                let product1: Mat<$t> =
                    <_ as ArgminDot<_, _>>::dot(&a, &b);
                let product2: Mat<$t> =
                    <_ as ArgminDot<_, _>>::dot(&a.as_mat_ref(), &b);
                let product3: Mat<$t> =
                    <_ as ArgminDot<_, _>>::dot(&a, &b.as_mat_ref());
                let product4: Mat<$t> =
                    <_ as ArgminDot<_, _>>::dot(&a.as_mat_ref(), &b.as_mat_ref());
                for i in 0..3 {
                    assert_relative_eq!(res[(i,0)] as f64, product1[(i,0)] as f64, epsilon = f64::EPSILON);
                    assert_relative_eq!(res[(i,0)] as f64, product2[(i,0)] as f64, epsilon = f64::EPSILON);
                    assert_relative_eq!(res[(i,0)] as f64, product3[(i,0)] as f64, epsilon = f64::EPSILON);
                    assert_relative_eq!(res[(i,0)] as f64, product4[(i,0)] as f64, epsilon = f64::EPSILON);
                }
            }
        }

        item! {
            #[test]
            fn [<test_mat_mat_ $t>]() {
                let a = matrix3_new(
                    1 as $t, 2 as $t, 3 as $t,
                    4 as $t, 5 as $t, 6 as $t,
                    3 as $t, 2 as $t, 1 as $t
                );
                let b = matrix3_new(
                    3 as $t, 2 as $t, 1 as $t,
                    6 as $t, 5 as $t, 4 as $t,
                    2 as $t, 4 as $t, 3 as $t
                );
                let res = matrix3_new(
                    21 as $t, 24 as $t, 18 as $t,
                    54 as $t, 57 as $t, 42 as $t,
                    23 as $t, 20 as $t, 14 as $t
                );
                let product1: Mat<$t> =
                    <_ as ArgminDot<_, _>>::dot(&a, &b);
                let product2: Mat<$t> =
                    <_ as ArgminDot<_, _>>::dot(&a.as_mat_ref(), &b);
                let product3: Mat<$t> =
                    <_ as ArgminDot<_, _>>::dot(&a, &b.as_mat_ref());
                let product4: Mat<$t> =
                    <_ as ArgminDot<_, _>>::dot(&a.as_mat_ref(), &b.as_mat_ref());
                for i in 0..3 {
                    for j in 0..3 {
                        assert_relative_eq!(res[(i, j)] as f64, product1[(i, j)] as f64, epsilon = f64::EPSILON);
                        assert_relative_eq!(res[(i, j)] as f64, product2[(i, j)] as f64, epsilon = f64::EPSILON);
                        assert_relative_eq!(res[(i, j)] as f64, product3[(i, j)] as f64, epsilon = f64::EPSILON);
                        assert_relative_eq!(res[(i, j)] as f64, product4[(i, j)] as f64, epsilon = f64::EPSILON);
                    }
                }
            }
        }

        item! {
            #[test]
            fn [<test_mat_primitive_ $t>]() {
                let a = matrix3_new(
                    1 as $t, 2 as $t, 3 as $t,
                    4 as $t, 5 as $t, 6 as $t,
                    3 as $t, 2 as $t, 1 as $t
                );
                let res = matrix3_new(
                    2 as $t, 4 as $t, 6 as $t,
                    8 as $t, 10 as $t, 12 as $t,
                    6 as $t, 4 as $t, 2 as $t
                );
                let product1: Mat<$t> =
                    <_ as ArgminDot<$t, _>>::dot(&a, &(2 as $t));
                let product2: Mat<$t> =
                    <_ as ArgminDot<$t, _>>::dot(&a.as_mat_ref(), &(2 as $t));
                assert_eq!(product1, product2);
                assert_eq!(product1.nrows(), 3);
                assert_eq!(product1.ncols(), 3);
                for i in 0..3 {
                    for j in 0..3 {
                        assert_relative_eq!(res[(i, j)] as f64, product1[(i, j)] as f64, epsilon = f64::EPSILON);
                    }
                }
            }
        }

        item! {
            #[test]
            fn [<test_primitive_mat_ $t>]() {
                let a = matrix3_new(
                    1 as $t, 2 as $t, 3 as $t,
                    4 as $t, 5 as $t, 6 as $t,
                    3 as $t, 2 as $t, 1 as $t
                );
                let res = matrix3_new(
                    2 as $t, 4 as $t, 6 as $t,
                    8 as $t, 10 as $t, 12 as $t,
                    6 as $t, 4 as $t, 2 as $t
                );
                let product1: Mat<$t> =
                    <$t as ArgminDot<_, _>>::dot(&(2 as $t), &a);
                let product2: Mat<$t> =
                    <$t as ArgminDot<_, _>>::dot(&(2 as $t), &a.as_mat_ref());
                assert_eq!(product1, product2);
                assert_eq!(product1.nrows(), 3);
                assert_eq!(product1.ncols(), 3);
                for i in 0..3 {
                    for j in 0..3 {
                        assert_relative_eq!(res[(i, j)] as f64, product1[(i, j)] as f64, epsilon = f64::EPSILON);
                    }
                }
            }
        }
    };
}

make_test!(f32);
make_test!(f64);
