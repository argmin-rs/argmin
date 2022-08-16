// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::{ArgminAxisIter, ArgminIter, ArgminMutIter};

macro_rules! make_iter {
    ($t:ty) => {
        impl ArgminIter<$t> for Vec<$t> {
            #[inline]
            fn iterator<'a>(&'a self) -> Box<dyn Iterator<Item = &'a $t> + 'a> {
                Box::new(self.iter())
            }
        }

        impl ArgminMutIter<$t> for Vec<$t> {
            #[inline]
            fn iterator_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut $t> + 'a> {
                Box::new(self.iter_mut())
            }
        }

        impl ArgminIter<$t> for Vec<Vec<$t>> {
            #[inline]
            fn iterator<'a>(&'a self) -> Box<dyn Iterator<Item = &'a $t> + 'a> {
                Box::new(self.iter().flatten())
            }
        }

        impl ArgminMutIter<Vec<$t>> for Vec<Vec<$t>> {
            fn iterator_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut Vec<$t>> + 'a> {
                Box::new(self.iter_mut())
            }
        }

        impl ArgminAxisIter<Vec<$t>> for Vec<Vec<$t>> {
            #[inline]
            fn row_iterator<'a>(&'a self) -> Box<dyn Iterator<Item = Vec<$t>> + 'a> {
                Box::new((0..self.len()).map(|i| (0..self[0].len()).map(|j| self[i][j]).collect()))
            }
            #[inline]
            fn column_iterator<'a>(&'a self) -> Box<dyn Iterator<Item = Vec<$t>> + 'a> {
                Box::new((0..self[0].len()).map(|j| (0..self.len()).map(|i| self[i][j]).collect()))
            }
        }
    };
}

make_iter!(i32);
make_iter!(i64);
make_iter!(f32);
make_iter!(f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iter() {
        assert_eq!(
            ArgminIter::iterator(&vec![1i32, 2, 3])
                .map(|&e| e * 2)
                .collect::<Vec<i32>>(),
            vec![2i32, 4, 6]
        );
        let mut v = vec![1i32, 2, 3];
        ArgminMutIter::iterator_mut(&mut v).for_each(|e| *e *= 2);
        assert_eq!(v, vec![2i32, 4, 6]);
    }

    #[test]
    fn test_axis_iter() {
        let m = vec![vec![1, 2, 3], vec![4, 5, 6]];

        let row_sums: Vec<i32> = m.row_iterator().map(|v: Vec<i32>| v.iter().sum()).collect();
        let col_sums: Vec<i32> = m
            .column_iterator()
            .map(|v: Vec<i32>| v.iter().sum())
            .collect();

        assert_eq!(row_sums, vec![6, 15]);
        assert_eq!(col_sums, vec![5, 7, 9]);
    }
}
