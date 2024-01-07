// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use rand::{distributions::uniform::SampleUniform, Rng};

use crate::ArgminRandom;

use nalgebra::{
    base::{allocator::Allocator, dimension::Dim, Scalar},
    DefaultAllocator, OMatrix,
};

impl<N, R, C> ArgminRandom for OMatrix<N, R, C>
where
    N: Scalar + PartialOrd + SampleUniform,
    R: Dim,
    C: Dim,
    DefaultAllocator: Allocator<N, R, C>,
{
    #[inline]
    fn rand_from_range<T: Rng>(min: &Self, max: &Self, rng: &mut T) -> OMatrix<N, R, C> {
        assert!(!min.is_empty());
        assert_eq!(min.shape(), max.shape());

        Self::from_iterator_generic(
            R::from_usize(min.nrows()),
            C::from_usize(min.ncols()),
            min.iter().zip(max.iter()).map(|(a, b)| {
                // Do not require a < b:

                // We do want to know if a and b are *exactly* the same.
                #[allow(clippy::float_cmp)]
                if a == b {
                    a.clone()
                } else if a < b {
                    rng.gen_range(a.clone()..b.clone())
                } else {
                    rng.gen_range(b.clone()..a.clone())
                }
            }),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix2x3, Vector3};
    use paste::item;
    use rand::SeedableRng;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_random_vec_ $t>]() {
                    let a = Vector3::new(1 as $t, 2 as $t, 3 as $t);
                    let b = Vector3::new(2 as $t, 3 as $t, 4 as $t);
                    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
                    let random = Vector3::<$t>::rand_from_range(&a, &b, &mut rng);
                    for i in 0..3 {
                        assert!(random[i] >= a[i]);
                        assert!(random[i] <= b[i]);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_random_vec_equal $t>]() {
                    let a = Vector3::new(1 as $t, 2 as $t, 3 as $t);
                    let b = Vector3::new(1 as $t, 2 as $t, 3 as $t);
                    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
                    let random = Vector3::<$t>::rand_from_range(&a, &b, &mut rng);
                    for i in 0..3 {
                        assert!((random[i] as f64 - a[i] as f64).abs() < std::f64::EPSILON);
                        assert!((random[i] as f64 - b[i] as f64).abs() < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_random_vec_reverse_ $t>]() {
                    let b = Vector3::new(1 as $t, 2 as $t, 3 as $t);
                    let a = Vector3::new(2 as $t, 3 as $t, 4 as $t);
                    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
                    let random = Vector3::<$t>::rand_from_range(&a, &b, &mut rng);
                    for i in 0..3 {
                        assert!(random[i] >= b[i]);
                        assert!(random[i] <= a[i]);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_random_mat_ $t>]() {
                    let a = Matrix2x3::new(
                        1 as $t, 3 as $t, 5 as $t,
                        2 as $t, 4 as $t, 6 as $t
                    );
                    let b = Matrix2x3::new(
                        2 as $t, 4 as $t, 6 as $t,
                        3 as $t, 5 as $t, 7 as $t
                    );
                    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
                    let random = Matrix2x3::<$t>::rand_from_range(&a, &b, &mut rng);
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

    make_test!(i8);
    make_test!(u8);
    make_test!(i16);
    make_test!(u16);
    make_test!(i32);
    make_test!(u32);
    make_test!(i64);
    make_test!(u64);
    make_test!(isize);
    make_test!(usize);
    make_test!(f32);
    make_test!(f64);
}
