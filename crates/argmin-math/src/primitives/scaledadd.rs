// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::{ArgminAdd, ArgminMul, ArgminScaledAdd};

// This is a very generic implementation. Once the specialization feature is stable, impls for
// types, which allow efficient execution of scaled adds can be made.
impl<T, U, W> ArgminScaledAdd<T, U, W> for W
where
    U: ArgminMul<T, T>,
    W: ArgminAdd<T, W>,
{
    #[inline]
    fn scaled_add(&self, factor: &U, vec: &T) -> W {
        self.add(&factor.mul(vec))
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
                fn [<test_scaledadd_ $t>]() {
                    let a = 2 as $t;
                    let b = 4 as $t;
                    let c = 10 as $t;
                    let res = <$t as ArgminScaledAdd<$t, $t, $t>>::scaled_add(&a, &b, &c);
                    assert_relative_eq!(42 as f64, res as f64, epsilon = std::f64::EPSILON);
                }
            }
        };
    }

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
