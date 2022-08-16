// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminSize;

macro_rules! make_size {
    ($t:ty) => {
        impl ArgminSize<usize> for Vec<$t> {
            #[inline]
            fn shape(&self) -> usize {
                self.len()
            }
        }
    };
}

make_size!(isize);
make_size!(usize);
make_size!(i8);
make_size!(i16);
make_size!(i32);
make_size!(i64);
make_size!(u8);
make_size!(u16);
make_size!(u32);
make_size!(u64);
make_size!(f32);
make_size!(f64);
