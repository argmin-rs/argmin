use faer::{unzipped, Entity, Mat};
use rand::distributions::uniform::SampleUniform;

use crate::ArgminRandom;

impl<E: Entity + PartialOrd + SampleUniform> ArgminRandom for Mat<E> {
    fn rand_from_range<R: rand::Rng>(min: &Self, max: &Self, rng: &mut R) -> Self {
        assert!(
            min.nrows() != 0 || min.ncols() != 0,
            "internal error: empty matrix unexpected"
        );
        assert_eq!(
            min.shape(),
            max.shape(),
            "internal error: matrices of same shape expected"
        );

        faer::zipped_rw!(min, max).map(|unzipped!(min, max)| {
            let a = min.read();
            let b = max.read();
            //@note(geo-ant) code was copied from the nalgebra implementation
            // to get the exact same behaviour
            // Do not require a < b:
            // We do want to know if a and b are *exactly* the same.
            #[allow(clippy::float_cmp)]
            if a == b {
                a
            } else if a < b {
                rng.gen_range(a..b)
            } else {
                rng.gen_range(b..a)
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helper::*;
    use super::*;
    use faer::mat;
    use faer::Mat;
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
}
