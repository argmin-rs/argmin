// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminFromSlice;

macro_rules! make_fromslice {
    ($t:ty) => {
        impl ArgminFromSlice<$t> for Vec<$t> {
            #[inline]
            fn from_slice(s: &[$t]) -> Self {
                s.to_vec()
            }
        }
    };
}

make_fromslice!(f32);
make_fromslice!(f64);
make_fromslice!(i8);
make_fromslice!(i16);
make_fromslice!(i32);
make_fromslice!(i64);
make_fromslice!(u8);
make_fromslice!(u16);
make_fromslice!(u32);
make_fromslice!(u64);
