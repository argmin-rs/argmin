// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminFromSlice;
use faer::{Mat, SimpleEntity};

impl<E> ArgminFromSlice<E> for Mat<E>
where
    E: SimpleEntity,
{
    #[inline]
    fn from_slice(s: &[E]) -> Self {
        Mat::from_fn(s.len(), 1, |r, _| s[r])
    }
}
