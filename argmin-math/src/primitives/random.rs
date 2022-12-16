// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminRandom;
use rand::Rng;

macro_rules! make_random {
    ($t:ty) => {
        impl ArgminRandom for $t {
            #[inline]
            fn rand_from_range(min: &Self, max: &Self) -> $t {
                rand::thread_rng().gen_range(*min..*max)
            }
        }
    };
}

make_random!(f32);
make_random!(f64);
make_random!(i8);
make_random!(i16);
make_random!(i32);
make_random!(i64);
make_random!(u8);
make_random!(u16);
make_random!(u32);
make_random!(u64);
make_random!(isize);
make_random!(usize);

#[cfg(test)]
mod tests {
    use super::*;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_random_vec_ $t>]() {
                    let a = 1 as $t;
                    let b = 2 as $t;
                    let random = $t::rand_from_range(&a, &b);
                    assert!(random >= a);
                    assert!(random <= b);
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
    make_test!(isize);
    make_test!(usize);
}
