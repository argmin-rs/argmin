// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

#[cfg(test)]
mod tests {
    use crate::ArgminScaledSub;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_scaledsub_vec_ $t>]() {
                    let a = vec![10 as $t, 20 as $t, 30 as $t];
                    let b = 2 as $t;
                    let c = vec![4 as $t, 5 as $t, 6 as $t];
                    let res = a.scaled_sub(&b, &c);
                    let target = vec![2 as $t, 10 as $t, 18 as $t];
                    for i in 0..3 {
                        assert!((((res[i] - target[i]) as f64).abs()) < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_scaledsub_vec_panic_1_ $t>]() {
                    let a = vec![1 as $t, 2 as $t, 3 as $t];
                    let b = 2 as $t;
                    let c = vec![4 as $t, 5 as $t];
                    a.scaled_sub(&b, &c);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_scaledsub_vec_panic_2_ $t>]() {
                    let a = vec![1 as $t, 2 as $t];
                    let b = 2 as $t;
                    let c = vec![4 as $t, 5 as $t, 6 as $t];
                    a.scaled_sub(&b, &c);
                }
            }

            item! {
                #[test]
                fn [<test_scaledsub_vec_vec_ $t>]() {
                    let a = vec![15 as $t, 20 as $t, 30 as $t];
                    let b = vec![3 as $t, 2 as $t, 1 as $t];
                    let c = vec![4 as $t, 5 as $t, 6 as $t];
                    let res = a.scaled_sub(&b, &c);
                    let target = vec![3 as $t, 10 as $t, 24 as $t];
                    for i in 0..3 {
                        assert!((((res[i] - target[i]) as f64).abs()) < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_scaledsub_vec_vec_panic_1_ $t>]() {
                    let a = vec![1 as $t, 2 as $t];
                    let b = vec![3 as $t, 2 as $t, 1 as $t];
                    let c = vec![4 as $t, 5 as $t, 6 as $t];
                    a.scaled_sub(&b, &c);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_scaledsub_vec_vec_panic_2_ $t>]() {
                    let a = vec![1 as $t, 2 as $t, 3 as $t];
                    let b = vec![3 as $t, 2 as $t];
                    let c = vec![4 as $t, 5 as $t, 6 as $t];
                    a.scaled_sub(&b, &c);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_scaledsub_vec_vec_panic_3_ $t>]() {
                    let a = vec![1 as $t, 2 as $t, 3 as $t];
                    let b = vec![3 as $t, 2 as $t, 1 as $t];
                    let c = vec![4 as $t, 5 as $t];
                    a.scaled_sub(&b, &c);
                }
            }

            item! {
                #[test]
                fn [<test_scaledsub_mat_mat_ $t>]() {
                    let a = vec![
                        vec![10 as $t, 20 as $t],
                        vec![30 as $t, 40 as $t],
                    ];
                    let b = vec![
                        vec![4 as $t, 3 as $t],
                        vec![2 as $t, 1 as $t],
                    ];
                    let c = vec![
                        vec![1 as $t, 2 as $t],
                        vec![2 as $t, 1 as $t],
                    ];
                    let res = a.scaled_sub(&b, &c);
                    let target = vec![
                        vec![6 as $t, 14 as $t],
                        vec![26 as $t, 39 as $t],
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
                fn [<test_scaledsub_mat_mat_panic_1_ $t>]() {
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
                    a.scaled_sub(&b, &c);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_scaledsub_mat_mat_panic_2_ $t>]() {
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
                    a.scaled_sub(&b, &c);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_scaledsub_mat_mat_panic_3_ $t>]() {
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
                    a.scaled_sub(&b, &c);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_scaledsub_mat_mat_panic_4_ $t>]() {
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
                    a.scaled_sub(&b, &c);

                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_scaledsub_mat_mat_panic_5_ $t>]() {
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
                    a.scaled_sub(&b, &c);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_scaledsub_mat_mat_panic_6_ $t>]() {
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
                    a.scaled_sub(&b, &c);
                }
            }

            item! {
                #[test]
                fn [<test_scaledsub_mat_scalar_ $t>]() {
                    let a = vec![
                        vec![10 as $t, 20 as $t],
                        vec![30 as $t, 40 as $t],
                    ];
                    let b = 2 as $t;
                    let c = vec![
                        vec![1 as $t, 2 as $t],
                        vec![2 as $t, 1 as $t],
                    ];
                    let res = a.scaled_sub(&b, &c);
                    let target = vec![
                        vec![8 as $t, 16 as $t],
                        vec![26 as $t, 38 as $t],
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
