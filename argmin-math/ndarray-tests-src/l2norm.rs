// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;
    use argmin_math::ArgminL2Norm;

    use ndarray::{array, Array1};
    use num_complex::Complex;
    use num_integer::Roots;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_norm_ $t>]() {
                    let a = array![4 as $t, 3 as $t];
                    let res = <Array1<$t> as ArgminL2Norm<$t>>::l2_norm(&a);
                    let target = 5 as $t;
                    assert!(((target - res) as f64).abs() < std::f64::EPSILON);
                }
            }

            item! {
                #[test]
                fn [<test_norm_complex_ $t>]() {
                    let a = array![Complex::new(4 as $t, 2 as $t), Complex::new(3 as $t, 4 as $t)];
                    let res = <Array1<Complex<$t>> as ArgminL2Norm<$t>>::l2_norm(&a);
                    let target = (a[0].norm_sqr() + a[1].norm_sqr()).sqrt();
                    assert!(((target - res) as f64).abs() < std::f64::EPSILON);
                }
            }
        };
    }

    macro_rules! make_test_signed {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_norm_signed_ $t>]() {
                    let a = array![-4 as $t, -3 as $t];
                    let res = <Array1<$t> as ArgminL2Norm<$t>>::l2_norm(&a);
                    let target = 5 as $t;
                    assert!(((target - res) as f64).abs() < std::f64::EPSILON);
                }
            }

            item! {
                #[test]
                fn [<test_norm_signed_complex_ $t>]() {
                    let a = array![Complex::new(-4 as $t, -2 as $t), Complex::new(-3 as $t, -4 as $t)];
                    let res = <Array1<Complex<$t>> as ArgminL2Norm<$t>>::l2_norm(&a);
                    let target = (a[0].norm_sqr() + a[1].norm_sqr()).sqrt();
                    assert!(((target - res) as f64).abs() < std::f64::EPSILON);
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

    make_test_signed!(isize);
    make_test_signed!(i8);
    make_test_signed!(i16);
    make_test_signed!(i32);
    make_test_signed!(i64);
    make_test_signed!(f32);
    make_test_signed!(f64);
}
