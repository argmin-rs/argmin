// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminBroadcast;

macro_rules! make_add {
    ($t:ty) => {
        impl ArgminBroadcast<Vec<$t>, Vec<Vec<$t>>> for Vec<Vec<$t>> {
            #[inline]
            fn broadcast_add(&self, other: &Vec<$t>) -> Vec<Vec<$t>> {
                self.iter()
                    .map(|row| {
                        row.iter()
                            .zip(other)
                            .map(|(&v1, &v2)| v1 + v2)
                            .collect::<Vec<$t>>()
                    })
                    .collect()
            }

            fn broadcast_sub(&self, other: &Vec<$t>) -> Vec<Vec<$t>> {
                self.iter()
                    .map(|row| {
                        row.iter()
                            .zip(other)
                            .map(|(&v1, &v2)| v1 - v2)
                            .collect::<Vec<$t>>()
                    })
                    .collect()
            }

            fn broadcast_mul(&self, other: &Vec<$t>) -> Vec<Vec<$t>> {
                self.iter()
                    .map(|row| {
                        row.iter()
                            .zip(other)
                            .map(|(&v1, &v2)| v1 * v2)
                            .collect::<Vec<$t>>()
                    })
                    .collect()
            }

            fn broadcast_div(&self, other: &Vec<$t>) -> Vec<Vec<$t>> {
                self.iter()
                    .map(|row| {
                        row.iter()
                            .zip(other)
                            .map(|(&v1, &v2)| v1 / v2)
                            .collect::<Vec<$t>>()
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
                fn [<test_broadcast_add_sub_mul_div_row_vec_ $t>]() {
                    let a = vec![
                        vec![2 as $t, 4 as $t, 6 as $t],
                        vec![1 as $t, 2 as $t, 3 as $t]
                    ];
                    let b = vec![1 as $t, 2 as $t, 3 as $t];
                    let expected_add = vec![
                        vec![3 as $t, 6 as $t, 9 as $t],
                        vec![2 as $t, 4 as $t, 6 as $t]
                    ];
                    let expected_sub = vec![
                        vec![1 as $t, 2 as $t, 3 as $t],
                        vec![0 as $t, 0 as $t, 0 as $t]
                    ];
                    let expected_mul = vec![
                        vec![2 as $t, 8 as $t, 18 as $t],
                        vec![1 as $t, 4 as $t, 9 as $t]
                    ];
                    let expected_div = vec![
                        vec![2 as $t, 2 as $t, 2 as $t],
                        vec![1 as $t, 1 as $t, 1 as $t]
                    ];
                    let res_add = a.broadcast_add(&b);
                    let res_sub = a.broadcast_sub(&b);
                    let res_mul = a.broadcast_mul(&b);
                    let res_div = a.broadcast_div(&b);
                    for i in 0..2 {
                        for j in 0..3 {
                            assert!((((res_add[i][j] - expected_add[i][j]) as f64).abs()) < std::f64::EPSILON);
                            assert!((((res_sub[i][j] - expected_sub[i][j]) as f64).abs()) < std::f64::EPSILON);
                            assert!((((res_mul[i][j] - expected_mul[i][j]) as f64).abs()) < std::f64::EPSILON);
                            assert!((((res_div[i][j] - expected_div[i][j]) as f64).abs()) < std::f64::EPSILON);
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
