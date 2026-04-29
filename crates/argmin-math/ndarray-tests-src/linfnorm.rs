// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;
    use approx::assert_relative_eq;
    use argmin_math::ArgminLInfNorm;
    use ndarray::{array, Array1};
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_linfnorm_ $t>]() {
                    let a = array![3 as $t, 4 as $t, 1 as $t];
                    let res = <Array1<$t> as ArgminLInfNorm<$t>>::linf_norm(&a);
                    let target = 4 as $t;
                    assert_relative_eq!(target as f64, res as f64, epsilon = f64::EPSILON);
                }
            }
        };
    }

    macro_rules! make_test_signed {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_linfnorm_signed_ $t>]() {
                    let a = array![-4 as $t, -3 as $t, 2 as $t];
                    let res = <Array1<$t> as ArgminLInfNorm<$t>>::linf_norm(&a);
                    let target = 4 as $t;
                    assert_relative_eq!(target as f64, res as f64, epsilon = f64::EPSILON);
                }
            }
        };
    }

    make_test!(u8);
    make_test!(u16);
    make_test!(u32);
    make_test!(u64);
    make_test!(i8);
    make_test!(i16);
    make_test!(i32);
    make_test!(i64);
    make_test!(f32);
    make_test!(f64);

    make_test_signed!(i8);
    make_test_signed!(i16);
    make_test_signed!(i32);
    make_test_signed!(i64);
    make_test_signed!(f32);
    make_test_signed!(f64);
}
