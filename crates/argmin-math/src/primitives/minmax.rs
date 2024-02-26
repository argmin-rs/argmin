// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminMinMax;

macro_rules! make_minmax {
    ($t:ty) => {
        impl ArgminMinMax for $t {
            #[inline]
            fn min(x: &Self, y: &Self) -> $t {
                if x <= y {
                    *x
                } else {
                    *y
                }
            }

            fn max(x: &Self, y: &Self) -> $t {
                if x >= y {
                    *x
                } else {
                    *y
                }
            }
        }
    };
}

make_minmax!(f32);
make_minmax!(f64);
make_minmax!(i8);
make_minmax!(i16);
make_minmax!(i32);
make_minmax!(i64);
make_minmax!(u8);
make_minmax!(u16);
make_minmax!(u32);
make_minmax!(u64);

#[cfg(test)]
mod tests {
    use super::*;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_minmax_ $t>]() {
                    let x = 5 as $t;
                    let y = 10 as $t;
                    assert_eq!(<$t as ArgminMinMax>::min(&x, &y).to_ne_bytes(), x.to_ne_bytes());
                    assert_eq!(<$t as ArgminMinMax>::max(&x, &y).to_ne_bytes(), y.to_ne_bytes());
                    assert_eq!(<$t as ArgminMinMax>::min(&y, &x).to_ne_bytes(), x.to_ne_bytes());
                    assert_eq!(<$t as ArgminMinMax>::max(&y, &x).to_ne_bytes(), y.to_ne_bytes());
                }
            }
        };
    }

    make_test!(f32);
    make_test!(f64);
    make_test!(i8);
    make_test!(u8);
    make_test!(i16);
    make_test!(u16);
    make_test!(i32);
    make_test!(u32);
    make_test!(i64);
    make_test!(u64);
}
