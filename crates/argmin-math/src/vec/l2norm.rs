// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminL2Norm;
use num_complex::Complex;
use num_integer::Roots;

macro_rules! make_norm_float {
    ($t:ty) => {
        impl ArgminL2Norm<$t> for Vec<$t> {
            #[inline]
            fn l2_norm(&self) -> $t {
                self.iter().map(|a| a.powi(2)).sum::<$t>().sqrt()
            }
        }
    };
}

macro_rules! make_norm_integer {
    ($t:ty) => {
        impl ArgminL2Norm<$t> for Vec<$t> {
            #[inline]
            fn l2_norm(&self) -> $t {
                self.iter().map(|a| a.pow(2)).sum::<$t>().sqrt()
            }
        }
    };
}

macro_rules! make_norm_complex {
    ($i: ty, $t:ty) => {
        impl ArgminL2Norm<$t> for Vec<$i> {
            #[inline]
            fn l2_norm(&self) -> $t {
                self.iter().map(|a| a.norm_sqr()).sum::<$t>().sqrt()
            }
        }
    };
}

macro_rules! make_norm_unsigned {
    ($t:ty) => {
        impl ArgminL2Norm<$t> for Vec<$t> {
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
make_norm_integer!(i8);
make_norm_integer!(i16);
make_norm_integer!(i32);
make_norm_integer!(i64);
make_norm_float!(f32);
make_norm_float!(f64);
make_norm_complex!(Complex<i8>, i8);
make_norm_complex!(Complex<i16>, i16);
make_norm_complex!(Complex<i32>, i32);
make_norm_complex!(Complex<i64>, i64);
make_norm_complex!(Complex<u8>, u8);
make_norm_complex!(Complex<u16>, u16);
make_norm_complex!(Complex<u32>, u32);
make_norm_complex!(Complex<u64>, u64);
make_norm_complex!(Complex<f32>, f32);
make_norm_complex!(Complex<f64>, f64);

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_norm_ $t>]() {
                    let a = vec![4 as $t, 3 as $t];
                    let res = <Vec<$t> as ArgminL2Norm<$t>>::l2_norm(&a);
                    let target = 5 as $t;
                    assert_relative_eq!(target as f64, res as f64, epsilon = f64::EPSILON);
                }
            }

            item! {
                #[test]
                fn [<test_norm_complex_ $t>]() {
                    let a = vec![Complex::new(4 as $t, 2 as $t), Complex::new(3 as $t, 4 as $t)];
                    let res = <Vec<Complex<$t>> as ArgminL2Norm<$t>>::l2_norm(&a);
                    let target = (a[0].norm_sqr() + a[1].norm_sqr()).sqrt();
                    assert_relative_eq!(target as f64, res as f64, epsilon = f64::EPSILON);
                }
            }
        };
    }

    macro_rules! make_test_signed {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_norm_signed_ $t>]() {
                    let a = vec![-4 as $t, -3 as $t];
                    let res = <Vec<$t> as ArgminL2Norm<$t>>::l2_norm(&a);
                    let target = 5 as $t;
                    assert_relative_eq!(target as f64, res as f64, epsilon = f64::EPSILON);
                }
            }

            item! {
                #[test]
                fn [<test_norm_signed_complex_ $t>]() {
                    let a = vec![Complex::new(-4 as $t, -2 as $t), Complex::new(-3 as $t, -4 as $t)];
                    let res = <Vec<Complex<$t>> as ArgminL2Norm<$t>>::l2_norm(&a);
                    let target = (a[0].norm_sqr() + a[1].norm_sqr()).sqrt();
                    assert_relative_eq!(target as f64, res as f64, epsilon = f64::EPSILON);
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

    make_test_signed!(i8);
    make_test_signed!(i16);
    make_test_signed!(i32);
    make_test_signed!(i64);
    make_test_signed!(f32);
    make_test_signed!(f64);
}
