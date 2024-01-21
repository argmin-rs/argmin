// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;
    use approx::assert_relative_eq;
    use argmin_math::ArgminMinMax;
    use ndarray::array;
    use ndarray::{Array1, Array2};
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_minmax_vec_vec_ $t>]() {
                    let a = array![1 as $t, 4 as $t, 8 as $t];
                    let b = array![2 as $t, 3 as $t, 4 as $t];
                    let target_max = array![2 as $t, 4 as $t, 8 as $t];
                    let target_min = array![1 as $t, 3 as $t, 4 as $t];
                    let res_max = <Array1<$t> as ArgminMinMax>::max(&a, &b);
                    let res_min = <Array1<$t> as ArgminMinMax>::min(&a, &b);
                    for i in 0..3 {
                        assert_relative_eq!(target_max[i] as f64, res_max[i] as f64, epsilon = std::f64::EPSILON);
                        assert_relative_eq!(target_min[i] as f64, res_min[i] as f64, epsilon = std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_minmax_mat_mat_ $t>]() {
                    let a = array![
                        [1 as $t, 4 as $t, 8 as $t],
                        [2 as $t, 5 as $t, 9 as $t]
                    ];
                    let b = array![
                        [2 as $t, 3 as $t, 4 as $t],
                        [3 as $t, 4 as $t, 5 as $t]
                    ];
                    let target_max = array![
                        [2 as $t, 4 as $t, 8 as $t],
                        [3 as $t, 5 as $t, 9 as $t]
                    ];
                    let target_min = array![
                        [1 as $t, 3 as $t, 4 as $t],
                        [2 as $t, 4 as $t, 5 as $t]
                    ];
                    let res_max = <Array2<$t> as ArgminMinMax>::max(&a, &b);
                    let res_min = <Array2<$t> as ArgminMinMax>::min(&a, &b);
                    for i in 0..3 {
                        for j in 0..2 {
                            assert_relative_eq!(target_max[(j, i)] as f64, res_max[(j, i)] as f64, epsilon = std::f64::EPSILON);
                            assert_relative_eq!(target_min[(j, i)] as f64, res_min[(j, i)] as f64, epsilon = std::f64::EPSILON);
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
