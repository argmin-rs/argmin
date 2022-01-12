// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

#[cfg(test)]
mod tests {
    use crate::ArgminScaledAdd;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_scaledadd_vec_ $t>]() {
                    let a = vec![1 as $t, 2 as $t, 3 as $t];
                    let b = 2 as $t;
                    let c = vec![4 as $t, 5 as $t, 6 as $t];
                    let res = a.scaled_add(&b, &c);
                    let target = vec![9 as $t, 12 as $t, 15 as $t];
                    for i in 0..3 {
                        assert!((((res[i] - target[i]) as f64).abs()) < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_scaledadd_vec_panic_1_ $t>]() {
                    let a = vec![1 as $t, 2 as $t, 3 as $t];
                    let b = 2 as $t;
                    let c = vec![4 as $t, 5 as $t];
                    a.scaled_add(&b, &c);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_scaledadd_vec_panic_2_ $t>]() {
                    let a = vec![1 as $t, 2 as $t];
                    let b = 2 as $t;
                    let c = vec![4 as $t, 5 as $t, 6 as $t];
                    a.scaled_add(&b, &c);
                }
            }

            item! {
                #[test]
                fn [<test_scaledadd_vec_vec_ $t>]() {
                    let a = vec![1 as $t, 2 as $t, 3 as $t];
                    let b = vec![3 as $t, 2 as $t, 1 as $t];
                    let c = vec![4 as $t, 5 as $t, 6 as $t];
                    let res = a.scaled_add(&b, &c);
                    let target = vec![13 as $t, 12 as $t, 9 as $t];
                    for i in 0..3 {
                        assert!((((res[i] - target[i]) as f64).abs()) < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_scaledadd_vec_vec_panic_1_ $t>]() {
                    let a = vec![1 as $t, 2 as $t];
                    let b = vec![3 as $t, 2 as $t, 1 as $t];
                    let c = vec![4 as $t, 5 as $t, 6 as $t];
                    a.scaled_add(&b, &c);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_scaledadd_vec_vec_panic_2_ $t>]() {
                    let a = vec![1 as $t, 2 as $t, 3 as $t];
                    let b = vec![3 as $t, 2 as $t];
                    let c = vec![4 as $t, 5 as $t, 6 as $t];
                    a.scaled_add(&b, &c);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_scaledadd_vec_vec_panic_3_ $t>]() {
                    let a = vec![1 as $t, 2 as $t, 3 as $t];
                    let b = vec![3 as $t, 2 as $t, 1 as $t];
                    let c = vec![4 as $t, 5 as $t];
                    a.scaled_add(&b, &c);
                }
            }

            item! {
                #[test]
                fn [<test_scaledadd_mat_mat_ $t>]() {
                    let a = vec![
                        vec![1 as $t, 2 as $t],
                        vec![3 as $t, 4 as $t],
                    ];
                    let b = vec![
                        vec![4 as $t, 3 as $t],
                        vec![2 as $t, 1 as $t],
                    ];
                    let c = vec![
                        vec![1 as $t, 2 as $t],
                        vec![2 as $t, 1 as $t],
                    ];
                    let res = a.scaled_add(&b, &c);
                    let target = vec![
                        vec![5 as $t, 8 as $t],
                        vec![7 as $t, 5 as $t],
                    ];
                    for i in 0..2 {
                        for j in 0..2 {
                            assert!((((res[i][j] - target[i][j]) as f64).abs()) < std::f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_scaledadd_mat_mat_panic_1_ $t>]() {
                    let a = vec![
                        vec![1 as $t],
                        vec![3 as $t, 4 as $t],
                    ];
                    let b = vec![
                        vec![4 as $t, 3 as $t],
                        vec![2 as $t, 1 as $t],
                    ];
                    let c = vec![
                        vec![1 as $t, 2 as $t],
                        vec![2 as $t, 1 as $t],
                    ];
                    a.scaled_add(&b, &c);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_scaledadd_mat_mat_panic_2_ $t>]() {
                    let a = vec![
                        vec![1 as $t, 2 as $t],
                        vec![3 as $t, 4 as $t],
                    ];
                    let b = vec![
                        vec![4 as $t, 3 as $t],
                        vec![1 as $t],
                    ];
                    let c = vec![
                        vec![1 as $t, 2 as $t],
                        vec![2 as $t, 1 as $t],
                    ];
                    a.scaled_add(&b, &c);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_scaledadd_mat_mat_panic_3_ $t>]() {
                    let a = vec![
                        vec![1 as $t, 2 as $t],
                        vec![3 as $t, 4 as $t],
                    ];
                    let b = vec![
                        vec![4 as $t, 3 as $t],
                        vec![2 as $t, 1 as $t],
                    ];
                    let c = vec![
                        vec![1 as $t],
                        vec![2 as $t, 1 as $t],
                    ];
                    a.scaled_add(&b, &c);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_scaledadd_mat_mat_panic_4_ $t>]() {
                    let a = vec![
                        vec![1 as $t, 2 as $t],
                    ];
                    let b = vec![
                        vec![4 as $t, 3 as $t],
                        vec![2 as $t, 1 as $t],
                    ];
                    let c = vec![
                        vec![1 as $t, 2 as $t],
                        vec![2 as $t, 1 as $t],
                    ];
                    a.scaled_add(&b, &c);

                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_scaledadd_mat_mat_panic_5_ $t>]() {
                    let a = vec![
                        vec![1 as $t, 2 as $t],
                        vec![3 as $t, 4 as $t],
                    ];
                    let b = vec![
                        vec![4 as $t, 3 as $t],
                    ];
                    let c = vec![
                        vec![1 as $t, 2 as $t],
                        vec![2 as $t, 1 as $t],
                    ];
                    a.scaled_add(&b, &c);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_scaledadd_mat_mat_panic_6_ $t>]() {
                    let a = vec![
                        vec![1 as $t, 2 as $t],
                        vec![3 as $t, 4 as $t],
                    ];
                    let b = vec![
                        vec![4 as $t, 3 as $t],
                        vec![2 as $t, 1 as $t],
                    ];
                    let c = vec![
                        vec![2 as $t, 1 as $t],
                    ];
                    a.scaled_add(&b, &c);
                }
            }

            item! {
                #[test]
                fn [<test_scaledadd_mat_scalar_ $t>]() {
                    let a = vec![
                        vec![1 as $t, 2 as $t],
                        vec![3 as $t, 4 as $t],
                    ];
                    let b = 2 as $t;
                    let c = vec![
                        vec![1 as $t, 2 as $t],
                        vec![2 as $t, 1 as $t],
                    ];
                    let res = a.scaled_add(&b, &c);
                    let target = vec![
                        vec![3 as $t, 6 as $t],
                        vec![7 as $t, 6 as $t],
                    ];
                    for i in 0..2 {
                        for j in 0..2 {
                            assert!((((res[i][j] - target[i][j]) as f64).abs()) < std::f64::EPSILON);
                        }
                    }
                }
            }
        };
    }

    make_test!(isize);
    make_test!(usize);
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
