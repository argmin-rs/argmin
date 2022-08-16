// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminDiv;
use num_complex::Complex;

macro_rules! make_div {
    ($t:ty) => {
        impl ArgminDiv<$t, Vec<$t>> for Vec<$t> {
            #[inline]
            fn div(&self, other: &$t) -> Vec<$t> {
                self.iter().map(|a| a / other).collect()
            }
        }

        impl ArgminDiv<Vec<$t>, Vec<$t>> for $t {
            #[inline]
            fn div(&self, other: &Vec<$t>) -> Vec<$t> {
                other.iter().map(|a| self / a).collect()
            }
        }

        impl ArgminDiv<Vec<$t>, Vec<$t>> for Vec<$t> {
            #[inline]
            fn div(&self, other: &Vec<$t>) -> Vec<$t> {
                let n1 = self.len();
                let n2 = other.len();
                assert!(n1 > 0);
                assert!(n2 > 0);
                assert_eq!(n1, n2);
                self.iter().zip(other.iter()).map(|(a, b)| a / b).collect()
            }
        }

        impl ArgminDiv<Vec<Vec<$t>>, Vec<Vec<$t>>> for Vec<Vec<$t>> {
            #[inline]
            fn div(&self, other: &Vec<Vec<$t>>) -> Vec<Vec<$t>> {
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
                        <Vec<$t> as ArgminDiv<Vec<$t>, Vec<$t>>>::div(&a, &b)
                    })
                    .collect()
            }
        }

        impl ArgminDiv<$t, Vec<Vec<$t>>> for Vec<Vec<$t>> {
            #[inline]
            fn div(&self, other: &$t) -> Vec<Vec<$t>> {
                self.iter().map(|a| a.div(other)).collect()
            }
        }
    };
}

make_div!(isize);
make_div!(usize);
make_div!(i8);
make_div!(u8);
make_div!(i16);
make_div!(u16);
make_div!(i32);
make_div!(u32);
make_div!(i64);
make_div!(u64);
make_div!(f32);
make_div!(f64);
make_div!(Complex<isize>);
make_div!(Complex<usize>);
make_div!(Complex<i8>);
make_div!(Complex<u8>);
make_div!(Complex<i16>);
make_div!(Complex<u16>);
make_div!(Complex<i32>);
make_div!(Complex<u32>);
make_div!(Complex<i64>);
make_div!(Complex<u64>);
make_div!(Complex<f32>);
make_div!(Complex<f64>);

#[cfg(test)]
mod tests {
    use super::*;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_div_vec_scalar_ $t>]() {
                    let a = vec![2 as $t, 4 as $t, 8 as $t];
                    let b = 2 as $t;
                    let target = vec![1 as $t, 2 as $t, 4 as $t];
                    let res = <Vec<$t> as ArgminDiv<$t, Vec<$t>>>::div(&a, &b);
                    for i in 0..3 {
                        assert!(((target[i] - res[i]) as f64).abs() < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_div_scalar_vec_ $t>]() {
                    let a = vec![2 as $t, 4 as $t, 8 as $t];
                    let b = 64 as $t;
                    let target = vec![32 as $t, 16 as $t, 8 as $t];
                    let res = <$t as ArgminDiv<Vec<$t>, Vec<$t>>>::div(&b, &a);
                    for i in 0..3 {
                        assert!(((target[i] - res[i]) as f64).abs() < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_div_vec_vec_ $t>]() {
                    let a = vec![4 as $t, 9 as $t, 8 as $t];
                    let b = vec![2 as $t, 3 as $t, 4 as $t];
                    let target = vec![2 as $t, 3 as $t, 2 as $t];
                    let res = <Vec<$t> as ArgminDiv<Vec<$t>, Vec<$t>>>::div(&a, &b);
                    for i in 0..3 {
                        assert!(((target[i] - res[i]) as f64).abs() < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_div_vec_vec_panic_ $t>]() {
                    let a = vec![1 as $t, 4 as $t];
                    let b = vec![41 as $t, 38 as $t, 34 as $t];
                    <Vec<$t> as ArgminDiv<Vec<$t>, Vec<$t>>>::div(&a, &b);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_div_vec_vec_panic_2_ $t>]() {
                    let a = vec![];
                    let b = vec![41 as $t, 38 as $t, 34 as $t];
                    <Vec<$t> as ArgminDiv<Vec<$t>, Vec<$t>>>::div(&a, &b);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_div_vec_vec_panic_3_ $t>]() {
                    let a = vec![41 as $t, 38 as $t, 34 as $t];
                    let b = vec![];
                    <Vec<$t> as ArgminDiv<Vec<$t>, Vec<$t>>>::div(&a, &b);
                }
            }

            item! {
                #[test]
                fn [<test_div_mat_mat_ $t>]() {
                    let a = vec![
                        vec![4 as $t, 12 as $t, 8 as $t],
                        vec![9 as $t, 20 as $t, 45 as $t]
                    ];
                    let b = vec![
                        vec![2 as $t, 3 as $t, 4 as $t],
                        vec![3 as $t, 4 as $t, 5 as $t]
                    ];
                    let target = vec![
                        vec![2 as $t, 4 as $t, 2 as $t],
                        vec![3 as $t, 5 as $t, 9 as $t]
                    ];
                    let res = <Vec<Vec<$t>> as ArgminDiv<Vec<Vec<$t>>, Vec<Vec<$t>>>>::div(&a, &b);
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
                fn [<test_div_mat_mat_panic_1_ $t>]() {
                    let a = vec![
                        vec![1 as $t, 4 as $t, 8 as $t],
                        vec![2 as $t, 9 as $t]
                    ];
                    let b = vec![
                        vec![41 as $t, 38 as $t, 34 as $t],
                        vec![40 as $t, 37 as $t, 33 as $t]
                    ];
                    <Vec<Vec<$t>> as ArgminDiv<Vec<Vec<$t>>, Vec<Vec<$t>>>>::div(&a, &b);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_div_mat_mat_panic_2_ $t>]() {
                    let a = vec![
                        vec![1 as $t, 4 as $t, 8 as $t],
                        vec![2 as $t, 5 as $t, 9 as $t]
                    ];
                    let b = vec![
                        vec![41 as $t, 38 as $t, 34 as $t],
                    ];
                    <Vec<Vec<$t>> as ArgminDiv<Vec<Vec<$t>>, Vec<Vec<$t>>>>::div(&a, &b);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_div_mat_mat_panic_3_ $t>]() {
                    let a = vec![
                        vec![1 as $t, 4 as $t, 8 as $t],
                        vec![2 as $t, 5 as $t, 9 as $t]
                    ];
                    let b = vec![];
                    <Vec<Vec<$t>> as ArgminDiv<Vec<Vec<$t>>, Vec<Vec<$t>>>>::div(&a, &b);
                }
            }

            item! {
                #[test]
                fn [<test_div_scalar_mat_1_ $t>]() {
                    let a = vec![
                        vec![16 as $t, 12 as $t, 10 as $t],
                        vec![8 as $t, 4 as $t, 2 as $t]
                    ];
                    let b = 2 as $t;
                    let target = vec![
                        vec![8 as $t, 6 as $t, 5 as $t],
                        vec![4 as $t, 2 as $t, 1 as $t]
                    ];
                    let res = <Vec<Vec<$t>> as ArgminDiv<$t, Vec<Vec<$t>>>>::div(&a, &b);
                    for i in 0..3 {
                        for j in 0..2 {
                        assert!(((target[j][i] - res[j][i]) as f64).abs() < std::f64::EPSILON);
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
