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
    use argmin_math::ArgminAdd;
    use ndarray::array;
    use ndarray::{Array1, Array2};
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_add_vec_scalar_ $t>]() {
                    let a = array![1 as $t, 4 as $t, 8 as $t];
                    let b = 34 as $t;
                    let target = array![35 as $t, 38 as $t, 42 as $t];
                    let res = <Array1<$t> as ArgminAdd<$t, Array1<$t>>>::add(&a, &b);
                    for i in 0..3 {
                        assert_relative_eq!(target[i] as f64, res[i] as f64, epsilon = std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_add_scalar_vec_ $t>]() {
                    let a = array![1 as $t, 4 as $t, 8 as $t];
                    let b = 34 as $t;
                    let target = array![35 as $t, 38 as $t, 42 as $t];
                    let res = <$t as ArgminAdd<Array1<$t>, Array1<$t>>>::add(&b, &a);
                    for i in 0..3 {
                        assert_relative_eq!(target[i] as f64, res[i] as f64, epsilon = std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_add_vec_vec_ $t>]() {
                    let a = array![1 as $t, 4 as $t, 8 as $t];
                    let b = array![41 as $t, 38 as $t, 34 as $t];
                    let target = array![42 as $t, 42 as $t, 42 as $t];
                    let res = <Array1<$t> as ArgminAdd<Array1<$t>, Array1<$t>>>::add(&a, &b);
                    for i in 0..3 {
                        assert_relative_eq!(target[i] as f64, res[i] as f64, epsilon = std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_add_vec_vec_panic_ $t>]() {
                    let a = array![1 as $t, 4 as $t];
                    let b = array![41 as $t, 38 as $t, 34 as $t];
                    <Array1<$t> as ArgminAdd<Array1<$t>, Array1<$t>>>::add(&a, &b);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_add_vec_vec_panic_2_ $t>]() {
                    let a = array![];
                    let b = array![41 as $t, 38 as $t, 34 as $t];
                    <Array1<$t> as ArgminAdd<Array1<$t>, Array1<$t>>>::add(&a, &b);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_add_vec_vec_panic_3_ $t>]() {
                    let a = array![41 as $t, 38 as $t, 34 as $t];
                    let b = array![];
                    <Array1<$t> as ArgminAdd<Array1<$t>, Array1<$t>>>::add(&a, &b);
                }
            }

            item! {
                #[test]
                fn [<test_add_mat_mat_ $t>]() {
                    let a = array![
                        [1 as $t, 4 as $t, 8 as $t],
                        [2 as $t, 5 as $t, 9 as $t]
                    ];
                    let b = array![
                        [41 as $t, 38 as $t, 34 as $t],
                        [40 as $t, 37 as $t, 33 as $t]
                    ];
                    let target = array![
                        [42 as $t, 42 as $t, 42 as $t],
                        [42 as $t, 42 as $t, 42 as $t]
                    ];
                    let res = <Array2<$t> as ArgminAdd<Array2<$t>, Array2<$t>>>::add(&a, &b);
                    for i in 0..3 {
                        for j in 0..2 {
                        assert_relative_eq!(target[(j, i)] as f64, res[(j, i)] as f64, epsilon = std::f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_add_mat_scalar_ $t>]() {
                    let a = array![
                        [1 as $t, 4 as $t, 8 as $t],
                        [2 as $t, 5 as $t, 9 as $t]
                    ];
                    let b = 2 as $t;
                    let target = array![
                        [3 as $t, 6 as $t, 10 as $t],
                        [4 as $t, 7 as $t, 11 as $t]
                    ];
                    let res = <Array2<$t> as ArgminAdd<$t, Array2<$t>>>::add(&a, &b);
                    for i in 0..3 {
                        for j in 0..2 {
                        assert_relative_eq!(target[(j, i)] as f64, res[(j, i)] as f64, epsilon = std::f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_add_mat_mat_panic_2_ $t>]() {
                    let a = array![
                        [1 as $t, 4 as $t, 8 as $t],
                        [2 as $t, 5 as $t, 9 as $t]
                    ];
                    let b = array![
                        [41 as $t, 38 as $t],
                    ];
                    <Array2<$t> as ArgminAdd<Array2<$t>, Array2<$t>>>::add(&a, &b);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_add_mat_mat_panic_3_ $t>]() {
                    let a = array![
                        [1 as $t, 4 as $t, 8 as $t],
                        [2 as $t, 5 as $t, 9 as $t]
                    ];
                    let b = array![[]];
                    <Array2<$t> as ArgminAdd<Array2<$t>, Array2<$t>>>::add(&a, &b);
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
    make_test!(f32);
    make_test!(f64);
}
