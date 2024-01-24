// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminDot;
use crate::ArgminWeightedDot;

impl<T, U, V> ArgminWeightedDot<T, U, V> for T
where
    Self: ArgminDot<T, U>,
    V: ArgminDot<T, T>,
{
    #[inline]
    fn weighted_dot(&self, w: &V, v: &T) -> U {
        self.dot(&w.dot(v))
    }
}

#[cfg(feature = "vec")]
#[cfg(test)]
mod tests_vec {
    use super::*;
    use approx::assert_relative_eq;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_ $t>]() {
                    let a = vec![2 as $t, 1 as $t, 2 as $t];
                    let b = vec![1 as $t, 2 as $t, 1 as $t];
                    let w = vec![
                        vec![8 as $t, 1 as $t, 6 as $t],
                        vec![3 as $t, 5 as $t, 7 as $t],
                        vec![4 as $t, 9 as $t, 2 as $t],
                    ];
                    let res: $t = a.weighted_dot(&w, &b);
                    assert_relative_eq!(100 as f64, res as f64, epsilon = std::f64::EPSILON);
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

#[cfg(feature = "ndarray_all")]
#[cfg(test)]
mod tests_ndarray {
    use super::*;
    use ndarray::array;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_ $t>]() {
                    let a = array![2 as $t, 1 as $t, 2 as $t];
                    let b = array![1 as $t, 2 as $t, 1 as $t];
                    let w = array![
                        [8 as $t, 1 as $t, 6 as $t],
                        [3 as $t, 5 as $t, 7 as $t],
                        [4 as $t, 9 as $t, 2 as $t],
                    ];
                    let res: $t = a.weighted_dot(&w, &b);
                    assert!((((res - 100 as $t) as f64).abs()) < std::f64::EPSILON);
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

#[cfg(feature = "nalgebra_all")]
#[cfg(test)]
mod tests_nalgebra {
    use super::*;
    use nalgebra::{Matrix3, Vector3};
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_ $t>]() {
                    let a = Vector3::new(2 as $t, 1 as $t, 2 as $t);
                    let b = Vector3::new(1 as $t, 2 as $t, 1 as $t);
                    let w = Matrix3::new(
                        8 as $t, 1 as $t, 6 as $t,
                        3 as $t, 5 as $t, 7 as $t,
                        4 as $t, 9 as $t, 2 as $t,
                    );
                    let res: $t = a.weighted_dot(&w, &b);
                    assert!((((res - 100 as $t) as f64).abs()) < std::f64::EPSILON);
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
