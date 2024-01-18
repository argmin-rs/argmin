// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::{ArgminMul, ArgminScaledSub, ArgminSub};

// This is a very generic implementation. Once the specialization feature is stable, impls for
// types, which allow efficient execution of scaled subs can be made.
impl<T, U, W> ArgminScaledSub<T, U, W> for W
where
    U: ArgminMul<T, T>,
    W: ArgminSub<T, W>,
{
    #[inline]
    fn scaled_sub(&self, factor: &U, vec: &T) -> W {
        self.sub(&factor.mul(vec))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_scaledsub_ $t>]() {
                    let a = 100 as $t;
                    let b = 2 as $t;
                    let c = 29 as $t;
                    let res = <$t as ArgminScaledSub<$t, $t, $t>>::scaled_sub(&a, &b, &c);
                    assert_relative_eq!(42 as f64, res as f64, epsilon = std::f64::EPSILON);
                }
            }
        };
    }

    make_test!(isize);
    make_test!(usize);
    make_test!(i8);
    make_test!(u8);
    make_test!(i16);
    make_test!(u16);
    make_test!(i32);
    make_test!(u32);
    make_test!(i64);
    make_test!(u64);
    make_test!(f32);
    make_test!(f64);
}
