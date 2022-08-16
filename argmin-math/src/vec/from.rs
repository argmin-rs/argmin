// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminFrom;

macro_rules! make_from {
    ($t:ty) => {
        impl<'a> ArgminFrom<&'a $t, usize> for Vec<$t> {
            #[inline]
            fn from_iterator<I: Iterator<Item = &'a $t>>(len: usize, iter: I) -> Self {
                let mut v: Vec<$t> = Vec::with_capacity(len);
                iter.take(len).for_each(|i| v.push(*i));
                v
            }
        }

        impl ArgminFrom<$t, usize> for Vec<$t> {
            #[inline]
            fn from_iterator<I: Iterator<Item = $t>>(len: usize, iter: I) -> Self {
                let mut v: Vec<$t> = Vec::with_capacity(len);
                iter.take(len).for_each(|i| v.push(i));
                v
            }
        }

        impl ArgminFrom<$t, (usize, usize)> for Vec<Vec<$t>> {
            #[inline]
            fn from_iterator<I: Iterator<Item = $t>>(shape: (usize, usize), iter: I) -> Self {
                let (n, m) = shape;
                let mut iter_mut = iter;

                (0..n)
                    .map(|_| (0..m).map(|_| iter_mut.next().unwrap()).collect())
                    .collect()
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
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_from_array_ $t>]() {

                    let v: Vec<$t> = Vec::from_iterator(3usize, [1 as $t, 2 as $t, 3 as $t].iter());
                    assert_eq!(v, vec![1 as $t, 2 as $t, 3 as $t]);

                }

                #[test]
                fn [<test_from_range_ $t>]() {

                    let v: Vec<$t> = Vec::from_iterator(3usize, (1..4).map(|v| v as $t));
                    assert_eq!(v, vec![1 as $t, 2 as $t, 3 as $t]);

                }

                #[test]
                fn [<test_from_range_2_ $t>]() {

                    let v: Vec<Vec<$t>> = Vec::from_iterator((3usize, 2usize), (0..6).map(|v| v as $t));
                    assert_eq!(v, vec![vec![0 as $t, 1 as $t], vec![2 as $t, 3 as $t], vec![4 as $t, 5 as $t]]);

                }
            }

        };
    }

    make_test!(f32);
    make_test!(f64);
}
