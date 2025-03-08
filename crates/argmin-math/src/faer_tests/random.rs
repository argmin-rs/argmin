use crate::faer_tests::test_helper::*;
use crate::ArgminRandom;
use faer::mat;
use paste::item;
use rand::SeedableRng;

macro_rules! make_test {
    ($t:ty) => {
        item! {
            #[test]
            fn [<test_random_vec_ $t>]() {
                let a = column_vector_from_vec(vec![1 as $t, 2 as $t, 3 as $t]);
                let b = column_vector_from_vec(vec![2 as $t, 3 as $t, 4 as $t]);
                let mut rng = rand::rngs::StdRng::seed_from_u64(42);
                let random = <_ as ArgminRandom>::rand_from_range(&a, &b, &mut rng);
                for i in 0..3 {
                    assert!(random[(i,0)] >= a[(i,0)]);
                    assert!(random[(i,0)] <= b[(i,0)]);
                }
            }
        }

        item! {
            #[test]
            fn [<test_random_vec_equal $t>]() {
                let a = column_vector_from_vec(vec![1 as $t, 2 as $t, 3 as $t]);
                let b = column_vector_from_vec(vec![1 as $t, 2 as $t, 3 as $t]);
                let mut rng = rand::rngs::StdRng::seed_from_u64(42);
                let random = <_ as ArgminRandom>::rand_from_range(&a, &b, &mut rng);
                for i in 0..3 {
                    assert!((random[(i,0)] as f64 - a[(i,0)] as f64).abs() < f64::EPSILON);
                    assert!((random[(i,0)] as f64 - b[(i,0)] as f64).abs() < f64::EPSILON);
                }
            }
        }

        item! {
            #[test]
            fn [<test_random_vec_reverse_ $t>]() {
                let b = column_vector_from_vec(vec![1 as $t, 2 as $t, 3 as $t]);
                let a = column_vector_from_vec(vec![2 as $t, 3 as $t, 4 as $t]);
                let mut rng = rand::rngs::StdRng::seed_from_u64(42);
                let random = <_ as ArgminRandom>::rand_from_range(&a, &b, &mut rng);
                for i in 0..3 {
                    assert!(random[(i,0)] >= b[(i,0)]);
                    assert!(random[(i,0)] <= a[(i,0)]);
                }
            }
        }

        item! {
            #[test]
            fn [<test_random_mat_ $t>]() {
                let a = mat![
                    [1 as $t, 3 as $t, 5 as $t],
                    [2 as $t, 4 as $t, 6 as $t]
                ];
                let b = mat![
                    [2 as $t, 4 as $t, 6 as $t],
                    [3 as $t, 5 as $t, 7 as $t]
                ];
                let mut rng = rand::rngs::StdRng::seed_from_u64(42);
                let random = <_ as ArgminRandom>::rand_from_range(&a, &b, &mut rng);
                for i in 0..3 {
                    for j in 0..2 {
                        assert!(random[(j, i)] >= a[(j, i)]);
                        assert!(random[(j, i)] <= b[(j, i)]);
                    }
                }
            }
        }
    };
}

make_test!(f32);
make_test!(f64);
