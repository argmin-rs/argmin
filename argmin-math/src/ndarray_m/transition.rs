// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

// Note: This is not really the preferred way I think. Maybe this should also be implemented for
// ArrayViews, which would probably make it more efficient.

use crate::ArgminTransition;
use ndarray::{Array1, Array2};

macro_rules! define_transitions {
    ($t:ty) => {
        impl ArgminTransition for Array1<$t> {
            type Array1D = Self;
            type Array2D = Array2<$t>;
        }

        impl ArgminTransition for Array2<$t> {
            type Array1D = Array1<$t>;
            type Array2D = Self;
        }
    };
}

define_transitions!(i8);
define_transitions!(i16);
define_transitions!(i32);
define_transitions!(i64);
define_transitions!(u8);
define_transitions!(u16);
define_transitions!(u32);
define_transitions!(u64);
define_transitions!(f32);
define_transitions!(f64);
