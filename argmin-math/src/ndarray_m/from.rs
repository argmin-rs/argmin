// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminFrom;
use ndarray::{Array1, Array2};

macro_rules! make_from {
    ($t:ty) => {
        impl<'a> ArgminFrom<&'a $t, usize> for Array1<$t> {
            #[inline]
            fn from_iterator<I: Iterator<Item = &'a $t>>(len: usize, iter: I) -> Self {
                Array1::<$t>::from_iter(iter.take(len).map(|&v| v))
            }
        }

        impl ArgminFrom<$t, usize> for Array1<$t> {
            #[inline]
            fn from_iterator<I: Iterator<Item = $t>>(len: usize, iter: I) -> Self {
                Array1::<$t>::from_iter(iter.take(len))
            }
        }

        impl ArgminFrom<$t, (usize, usize)> for Array2<$t> {
            #[inline]
            fn from_iterator<I: Iterator<Item = $t>>(shape: (usize, usize), iter: I) -> Self {
                let mut iter = iter;
                Array2::<$t>::from_shape_fn(shape, |(_, _)| iter.next().unwrap())
            }
        }
    };
}

make_from!(i32);
make_from!(i64);
make_from!(f32);
make_from!(f64);

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, Array2};
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_from_array_ $t>]() {

                    let v: Array1<$t> = Array1::<$t>::from_iterator(3usize, [1 as $t, 2 as $t, 3 as $t].iter());
                    assert_eq!(v, array![1 as $t, 2 as $t, 3 as $t]);

                }

                #[test]
                fn [<test_from_range_ $t>]() {

                    let v: Array1<$t> = Array1::<$t>::from_iterator(3usize, (1..4).map(|v| v as $t));
                    assert_eq!(v, array![1 as $t, 2 as $t, 3 as $t]);

                }

                #[test]
                fn [<test_from_range_2_ $t>]() {

                    let v: Array2<$t> = Array2::<$t>::from_iterator((3usize, 2usize), (0..6).map(|v| v as $t));
                    assert_eq!(v, array![[0 as $t, 1 as $t], [2 as $t, 3 as $t], [4 as $t, 5 as $t]]);

                }
            }

        };
    }

    make_test!(f32);
    make_test!(f64);
}
