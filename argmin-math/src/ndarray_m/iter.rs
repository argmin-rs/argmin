// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::{ArgminAxisIter, ArgminIter, ArgminMutIter};
use ndarray::{Array1, Array2, Axis};

macro_rules! make_iter {
    ($t:ty) => {
        impl ArgminIter<$t> for Array1<$t> {
            #[inline]
            fn iterator<'a>(&'a self) -> Box<dyn Iterator<Item = &'a $t> + 'a> {
                Box::new(self.iter())
            }
        }

        impl ArgminMutIter<$t> for Array1<$t> {
            #[inline]
            fn iterator_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut $t> + 'a> {
                Box::new(self.iter_mut())
            }
        }

        impl ArgminIter<$t> for Array2<$t> {
            #[inline]
            fn iterator<'a>(&'a self) -> Box<dyn Iterator<Item = &'a $t> + 'a> {
                Box::new(self.iter())
            }
        }

        impl ArgminMutIter<$t> for Array2<$t> {
            #[inline]
            fn iterator_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut $t> + 'a> {
                Box::new(self.iter_mut())
            }
        }

        impl ArgminAxisIter<Array1<$t>> for Array2<$t> {
            #[inline]
            fn row_iterator<'a>(&'a self) -> Box<dyn Iterator<Item = Array1<$t>> + 'a> {
                Box::new(self.axis_iter(Axis(0)).map(|vec| vec.to_owned()))
            }

            #[inline]
            fn column_iterator<'a>(&'a self) -> Box<dyn Iterator<Item = Array1<$t>> + 'a> {
                Box::new(self.axis_iter(Axis(1)).map(|vec| vec.to_owned()))
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
    use ndarray::{array, Array1};

    #[test]
    fn test_iter() {
        assert_eq!(
            ArgminIter::iterator(&array![1i32, 2, 3])
                .map(|&e| e * 2)
                .collect::<Vec<i32>>(),
            vec![2i32, 4, 6]
        );
        let mut v = array![1i32, 2, 3];
        ArgminMutIter::iterator_mut(&mut v).for_each(|e| *e *= 2);
        assert_eq!(v, array![2i32, 4, 6]);
    }

    #[test]
    fn test_axis_iter() {
        let m = array![[1, 2, 3], [4, 5, 6]];

        let row_sums: Vec<i32> = m.row_iterator().map(|v: Array1<i32>| v.sum()).collect();
        let col_sums: Vec<i32> = m.column_iterator().map(|v: Array1<i32>| v.sum()).collect();

        assert_eq!(row_sums, vec![6, 15]);
        assert_eq!(col_sums, vec![5, 7, 9]);
    }
}
