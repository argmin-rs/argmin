// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminFromSlice;
use nalgebra::{DVector, Scalar};

impl<T> ArgminFromSlice<T> for DVector<T>
where
    T: Scalar + Copy,
{
    #[inline]
    fn from_slice(s: &[T]) -> Self {
        DVector::from_column_slice(s)
    }
}
