// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminMinMax;

macro_rules! make_minmax {
    ($t:ty) => {
        impl ArgminMinMax for Vec<$t> {
            fn min(x: &Self, y: &Self) -> Self {
                assert!(!x.is_empty());
                assert_eq!(x.len(), y.len());

                x.iter()
                    .zip(y.iter())
                    .map(|(a, b)| if a < b { a.clone() } else { b.clone() })
                    .collect()
            }

            fn max(x: &Self, y: &Self) -> Self {
                assert!(!x.is_empty());
                assert_eq!(x.len(), y.len());

                x.iter()
                    .zip(y.iter())
                    .map(|(a, b)| if a > b { a.clone() } else { b.clone() })
                    .collect()
            }
        }

        impl ArgminMinMax for Vec<Vec<$t>> {
            fn min(x: &Self, y: &Self) -> Self {
                assert!(!x.is_empty());
                assert_eq!(x.len(), y.len());

                x.iter()
                    .zip(y.iter())
                    .map(|(a, b)| <Vec<$t> as ArgminMinMax>::min(&a, &b))
                    .collect()
            }

            fn max(x: &Self, y: &Self) -> Self {
                assert!(!x.is_empty());
                assert_eq!(x.len(), y.len());

                x.iter()
                    .zip(y.iter())
                    .map(|(a, b)| <Vec<$t> as ArgminMinMax>::max(&a, &b))
                    .collect()
            }
        }
    };
}

make_minmax!(isize);
make_minmax!(usize);
make_minmax!(i8);
make_minmax!(u8);
make_minmax!(i16);
make_minmax!(u16);
make_minmax!(i32);
make_minmax!(u32);
make_minmax!(i64);
make_minmax!(u64);
make_minmax!(f32);
make_minmax!(f64);

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_minmax_vec_vec_ $t>]() {
                    let a = vec![1 as $t, 4 as $t, 8 as $t];
                    let b = vec![2 as $t, 3 as $t, 4 as $t];
                    let target_max = vec![2 as $t, 4 as $t, 8 as $t];
                    let target_min = vec![1 as $t, 3 as $t, 4 as $t];
                    let res_max = <Vec<$t> as ArgminMinMax>::max(&a, &b);
                    let res_min = <Vec<$t> as ArgminMinMax>::min(&a, &b);
                    for i in 0..3 {
                        assert_relative_eq!(target_max[i] as f64, res_max[i] as f64, epsilon = std::f64::EPSILON);
                        assert_relative_eq!(target_min[i] as f64, res_min[i] as f64, epsilon = std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_minmax_mat_mat_ $t>]() {
                    let a = vec![
                        vec![1 as $t, 4 as $t, 8 as $t],
                        vec![2 as $t, 5 as $t, 9 as $t]
                    ];
                    let b = vec![
                        vec![2 as $t, 3 as $t, 4 as $t],
                        vec![3 as $t, 4 as $t, 5 as $t]
                    ];
                    let target_max = vec![
                        vec![2 as $t, 4 as $t, 8 as $t],
                        vec![3 as $t, 5 as $t, 9 as $t]
                    ];
                    let target_min = vec![
                        vec![1 as $t, 3 as $t, 4 as $t],
                        vec![2 as $t, 4 as $t, 5 as $t]
                    ];
                    let res_max = <Vec<Vec<$t>> as ArgminMinMax>::max(&a, &b);
                    let res_min = <Vec<Vec<$t>> as ArgminMinMax>::min(&a, &b);
                    for i in 0..3 {
                        for j in 0..2 {
                        assert_relative_eq!(target_max[j][i] as f64, res_max[j][i] as f64, epsilon = std::f64::EPSILON);
                        assert_relative_eq!(target_min[j][i] as f64, res_min[j][i] as f64, epsilon = std::f64::EPSILON);
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
