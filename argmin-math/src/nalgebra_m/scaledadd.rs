// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

#[cfg(test)]
mod tests {
    use crate::ArgminScaledAdd;
    use nalgebra::{DVector, Matrix2, Vector3};
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_scaledadd_vec_ $t>]() {
                    let a = Vector3::new(1 as $t, 2 as $t, 3 as $t);
                    let b = 2 as $t;
                    let c = Vector3::new(4 as $t, 5 as $t, 6 as $t);
                    let res = <Vector3<$t> as ArgminScaledAdd<Vector3<$t>, $t, Vector3<$t>>>::scaled_add(&a, &b, &c);
                    let target = Vector3::new(9 as $t, 12 as $t, 15 as $t);
                    for i in 0..3 {
                        assert!((((res[i] - target[i]) as f64).abs()) < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_scaledadd_vec_panic_1_ $t>]() {
                    let a = DVector::from_vec(vec![1 as $t, 2 as $t, 3 as $t]);
                    let b = 2 as $t;
                    let c = DVector::from_vec(vec![4 as $t, 5 as $t]);
                    <DVector<$t> as ArgminScaledAdd<DVector<$t>, $t, DVector<$t>>>::scaled_add(&a, &b, &c);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_scaledadd_vec_panic_2_ $t>]() {
                    let a = DVector::from_vec(vec![1 as $t, 2 as $t]);
                    let b = 2 as $t;
                    let c = DVector::from_vec(vec![4 as $t, 5 as $t, 6 as $t]);
                    <DVector<$t> as ArgminScaledAdd<DVector<$t>, $t, DVector<$t>>>::scaled_add(&a, &b, &c);
                }
            }

            item! {
                #[test]
                fn [<test_scaledadd_vec_vec_ $t>]() {
                    let a = Vector3::new(1 as $t, 2 as $t, 3 as $t);
                    let b = Vector3::new(3 as $t, 2 as $t, 1 as $t);
                    let c = Vector3::new(4 as $t, 5 as $t, 6 as $t);
                    let res = <Vector3<$t> as ArgminScaledAdd<Vector3<$t>, Vector3<$t>, Vector3<$t>>>::scaled_add(&a, &b, &c);
                    let target = Vector3::new(13 as $t, 12 as $t, 9 as $t);
                    for i in 0..3 {
                        assert!((((res[i] - target[i]) as f64).abs()) < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_scaledadd_vec_vec_panic_1_ $t>]() {
                    let a = DVector::from_vec(vec![1 as $t, 2 as $t]);
                    let b = DVector::from_vec(vec![3 as $t, 2 as $t, 1 as $t]);
                    let c = DVector::from_vec(vec![4 as $t, 5 as $t, 6 as $t]);
                    <DVector<$t> as ArgminScaledAdd<DVector<$t>, DVector<$t>, DVector<$t>>>::scaled_add(&a, &b, &c);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_scaledadd_vec_vec_panic_2_ $t>]() {
                    let a = DVector::from_vec(vec![1 as $t, 2 as $t, 3 as $t]);
                    let b = DVector::from_vec(vec![3 as $t, 2 as $t]);
                    let c = DVector::from_vec(vec![4 as $t, 5 as $t, 6 as $t]);
                    <DVector<$t> as ArgminScaledAdd<DVector<$t>, DVector<$t>, DVector<$t>>>::scaled_add(&a, &b, &c);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_scaledadd_vec_vec_panic_3_ $t>]() {
                    let a = DVector::from_vec(vec![1 as $t, 2 as $t, 3 as $t]);
                    let b = DVector::from_vec(vec![3 as $t, 2 as $t, 1 as $t]);
                    let c = DVector::from_vec(vec![4 as $t, 5 as $t]);
                    <DVector<$t> as ArgminScaledAdd<DVector<$t>, DVector<$t>, DVector<$t>>>::scaled_add(&a, &b, &c);
                }
            }

            item! {
                #[test]
                fn [<test_scaledadd_mat_mat_ $t>]() {
                    let a = Matrix2::new(
                        1 as $t, 2 as $t,
                        3 as $t, 4 as $t,
                    );
                    let b = Matrix2::new(
                        4 as $t, 3 as $t,
                        2 as $t, 1 as $t,
                    );
                    let c = Matrix2::new(
                        1 as $t, 2 as $t,
                        2 as $t, 1 as $t,
                    );
                    let res = <Matrix2<$t> as ArgminScaledAdd<Matrix2<$t>, Matrix2<$t>, Matrix2<$t>>>::scaled_add(&a, &b, &c);
                    let target = Matrix2::new(
                        5 as $t, 8 as $t,
                        7 as $t, 5 as $t,
                    );
                    for i in 0..2 {
                        for j in 0..2 {
                            assert!((((res[(i, j)] - target[(i, j)]) as f64).abs()) < std::f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_scaledadd_mat_scalar_ $t>]() {
                    let a = Matrix2::new(
                        1 as $t, 2 as $t,
                        3 as $t, 4 as $t,
                    );
                    let b = 2 as $t;
                    let c = Matrix2::new(
                        1 as $t, 2 as $t,
                        2 as $t, 1 as $t,
                    );
                    let res = <Matrix2<$t> as ArgminScaledAdd<Matrix2<$t>, $t, Matrix2<$t>>>::scaled_add(&a, &b, &c);
                    let target = Matrix2::new(
                        3 as $t, 6 as $t,
                        7 as $t, 6 as $t,
                    );
                    for i in 0..2 {
                        for j in 0..2 {
                            assert!((((res[(i, j)] - target[(i, j)]) as f64).abs()) < std::f64::EPSILON);
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
    make_test!(f32);
    make_test!(f64);
}
