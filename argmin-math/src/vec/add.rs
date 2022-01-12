// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminAdd;

macro_rules! make_add {
    ($t:ty) => {
        impl ArgminAdd<$t, Vec<$t>> for Vec<$t> {
            #[inline]
            fn add(&self, other: &$t) -> Vec<$t> {
                self.iter().map(|a| a + other).collect()
            }
        }

        impl ArgminAdd<Vec<$t>, Vec<$t>> for $t {
            #[inline]
            fn add(&self, other: &Vec<$t>) -> Vec<$t> {
                other.iter().map(|a| a + self).collect()
            }
        }

        impl ArgminAdd<Vec<$t>, Vec<$t>> for Vec<$t> {
            #[inline]
            fn add(&self, other: &Vec<$t>) -> Vec<$t> {
                let n1 = self.len();
                let n2 = other.len();
                assert!(n1 > 0);
                assert!(n2 > 0);
                assert_eq!(n1, n2);
                self.iter().zip(other.iter()).map(|(a, b)| a + b).collect()
            }
        }

        impl ArgminAdd<Vec<Vec<$t>>, Vec<Vec<$t>>> for Vec<Vec<$t>> {
            #[inline]
            fn add(&self, other: &Vec<Vec<$t>>) -> Vec<Vec<$t>> {
                let sr = self.len();
                let or = other.len();
                assert!(sr > 0);
                // implicitly, or > 0
                assert_eq!(sr, or);
                let sc = self[0].len();
                self.iter()
                    .zip(other.iter())
                    .map(|(a, b)| {
                        assert_eq!(a.len(), sc);
                        assert_eq!(b.len(), sc);
                        <Vec<$t> as ArgminAdd<Vec<$t>, Vec<$t>>>::add(&a, &b)
                    })
                    .collect()
            }
        }

        impl ArgminAdd<$t, Vec<Vec<$t>>> for Vec<Vec<$t>> {
            #[inline]
            fn add(&self, other: &$t) -> Vec<Vec<$t>> {
                let sr = self.len();
                assert!(sr > 0);
                let sc = self[0].len();
                self.iter()
                    .map(|a| {
                        assert_eq!(a.len(), sc);
                        <Vec<$t> as ArgminAdd<$t, Vec<$t>>>::add(&a, &other)
                    })
                    .collect()
            }
        }
    };
}

make_add!(isize);
make_add!(usize);
make_add!(i8);
make_add!(i16);
make_add!(i32);
make_add!(i64);
make_add!(u8);
make_add!(u16);
make_add!(u32);
make_add!(u64);
make_add!(f32);
make_add!(f64);

#[cfg(test)]
mod tests {
    use super::*;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_add_vec_scalar_ $t>]() {
                    let a = vec![1 as $t, 4 as $t, 8 as $t];
                    let b = 34 as $t;
                    let target = vec![35 as $t, 38 as $t, 42 as $t];
                    let res = <Vec<$t> as ArgminAdd<$t, Vec<$t>>>::add(&a, &b);
                    for i in 0..3 {
                        assert!(((target[i] - res[i]) as f64).abs() < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_add_scalar_vec_ $t>]() {
                    let a = vec![1 as $t, 4 as $t, 8 as $t];
                    let b = 34 as $t;
                    let target = vec![35 as $t, 38 as $t, 42 as $t];
                    let res = <$t as ArgminAdd<Vec<$t>, Vec<$t>>>::add(&b, &a);
                    for i in 0..3 {
                        assert!(((target[i] - res[i]) as f64).abs() < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_add_vec_vec_ $t>]() {
                    let a = vec![1 as $t, 4 as $t, 8 as $t];
                    let b = vec![41 as $t, 38 as $t, 34 as $t];
                    let target = vec![42 as $t, 42 as $t, 42 as $t];
                    let res = <Vec<$t> as ArgminAdd<Vec<$t>, Vec<$t>>>::add(&a, &b);
                    for i in 0..3 {
                        assert!(((target[i] - res[i]) as f64).abs() < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_add_vec_vec_panic_ $t>]() {
                    let a = vec![1 as $t, 4 as $t];
                    let b = vec![41 as $t, 38 as $t, 34 as $t];
                    <Vec<$t> as ArgminAdd<Vec<$t>, Vec<$t>>>::add(&a, &b);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_add_vec_vec_panic_2_ $t>]() {
                    let a = vec![];
                    let b = vec![41 as $t, 38 as $t, 34 as $t];
                    <Vec<$t> as ArgminAdd<Vec<$t>, Vec<$t>>>::add(&a, &b);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_add_vec_vec_panic_3_ $t>]() {
                    let a = vec![41 as $t, 38 as $t, 34 as $t];
                    let b = vec![];
                    <Vec<$t> as ArgminAdd<Vec<$t>, Vec<$t>>>::add(&a, &b);
                }
            }

            item! {
                #[test]
                fn [<test_add_mat_mat_ $t>]() {
                    let a = vec![
                        vec![1 as $t, 4 as $t, 8 as $t],
                        vec![2 as $t, 5 as $t, 9 as $t]
                    ];
                    let b = vec![
                        vec![41 as $t, 38 as $t, 34 as $t],
                        vec![40 as $t, 37 as $t, 33 as $t]
                    ];
                    let target = vec![
                        vec![42 as $t, 42 as $t, 42 as $t],
                        vec![42 as $t, 42 as $t, 42 as $t]
                    ];
                    let res = <Vec<Vec<$t>> as ArgminAdd<Vec<Vec<$t>>, Vec<Vec<$t>>>>::add(&a, &b);
                    for i in 0..3 {
                        for j in 0..2 {
                        assert!(((target[j][i] - res[j][i]) as f64).abs() < std::f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_add_mat_scalar_ $t>]() {
                    let a = vec![
                        vec![1 as $t, 4 as $t, 8 as $t],
                        vec![2 as $t, 5 as $t, 9 as $t]
                    ];
                    let b = 2 as $t;
                    let target = vec![
                        vec![3 as $t, 6 as $t, 10 as $t],
                        vec![4 as $t, 7 as $t, 11 as $t]
                    ];
                    let res = <Vec<Vec<$t>> as ArgminAdd<$t, Vec<Vec<$t>>>>::add(&a, &b);
                    for i in 0..3 {
                        for j in 0..2 {
                        assert!(((target[j][i] - res[j][i]) as f64).abs() < std::f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_add_mat_mat_panic_1_ $t>]() {
                    let a = vec![
                        vec![1 as $t, 4 as $t, 8 as $t],
                        vec![2 as $t, 9 as $t]
                    ];
                    let b = vec![
                        vec![41 as $t, 38 as $t, 34 as $t],
                        vec![40 as $t, 37 as $t, 33 as $t]
                    ];
                    <Vec<Vec<$t>> as ArgminAdd<Vec<Vec<$t>>, Vec<Vec<$t>>>>::add(&a, &b);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_add_mat_mat_panic_2_ $t>]() {
                    let a = vec![
                        vec![1 as $t, 4 as $t, 8 as $t],
                        vec![2 as $t, 5 as $t, 9 as $t]
                    ];
                    let b = vec![
                        vec![41 as $t, 38 as $t, 34 as $t],
                    ];
                    <Vec<Vec<$t>> as ArgminAdd<Vec<Vec<$t>>, Vec<Vec<$t>>>>::add(&a, &b);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_add_mat_mat_panic_3_ $t>]() {
                    let a = vec![
                        vec![1 as $t, 4 as $t, 8 as $t],
                        vec![2 as $t, 5 as $t, 9 as $t]
                    ];
                    let b = vec![];
                    <Vec<Vec<$t>> as ArgminAdd<Vec<Vec<$t>>, Vec<Vec<$t>>>>::add(&a, &b);
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
