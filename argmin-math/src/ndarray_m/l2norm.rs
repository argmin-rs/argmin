// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminL2Norm;
use ndarray::Array1;
use num_complex::Complex;
use num_integer::Roots;

macro_rules! make_norm_float {
    ($t:ty) => {
        impl ArgminL2Norm<$t> for Array1<$t> {
            #[inline]
            fn l2_norm(&self) -> $t {
                self.iter().map(|a| a.powi(2)).sum::<$t>().sqrt()
            }
        }
    };
}

macro_rules! make_norm_integer {
    ($t:ty) => {
        impl ArgminL2Norm<$t> for Array1<$t> {
            #[inline]
            fn l2_norm(&self) -> $t {
                self.iter().map(|a| a.pow(2)).sum::<$t>().sqrt()
            }
        }
    };
}

macro_rules! make_norm_complex {
    ($i: ty, $t:ty) => {
        impl ArgminL2Norm<$t> for Array1<$i> {
            #[inline]
            fn l2_norm(&self) -> $t {
                self.iter().map(|a| a.norm_sqr()).sum::<$t>().sqrt()
            }
        }
    };
}

macro_rules! make_norm_unsigned {
    ($t:ty) => {
        impl ArgminL2Norm<$t> for Array1<$t> {
            #[inline]
            fn l2_norm(&self) -> $t {
                self.iter().map(|a| a.pow(2)).sum::<$t>().sqrt()
            }
        }
    };
}

make_norm_unsigned!(u8);
make_norm_unsigned!(u16);
make_norm_unsigned!(u32);
make_norm_unsigned!(u64);
make_norm_unsigned!(usize);
make_norm_integer!(i8);
make_norm_integer!(i16);
make_norm_integer!(i32);
make_norm_integer!(i64);
make_norm_integer!(isize);
make_norm_float!(f32);
make_norm_float!(f64);
make_norm_complex!(Complex<i8>, i8);
make_norm_complex!(Complex<i16>, i16);
make_norm_complex!(Complex<i32>, i32);
make_norm_complex!(Complex<i64>, i64);
make_norm_complex!(Complex<isize>, isize);
make_norm_complex!(Complex<u8>, u8);
make_norm_complex!(Complex<u16>, u16);
make_norm_complex!(Complex<u32>, u32);
make_norm_complex!(Complex<u64>, u64);
make_norm_complex!(Complex<usize>, usize);
make_norm_complex!(Complex<f32>, f32);
make_norm_complex!(Complex<f64>, f64);

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1};
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
