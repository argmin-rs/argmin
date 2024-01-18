// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminL1Norm;
use num_complex::Complex;

macro_rules! make_l1norm_unsigned {
    ($t:ty) => {
        impl ArgminL1Norm<$t> for $t {
            #[inline]
            fn l1_norm(&self) -> $t {
                *self
            }
        }
    };
}

macro_rules! make_l1norm {
    ($t:ty) => {
        impl ArgminL1Norm<$t> for $t {
            #[inline]
            fn l1_norm(&self) -> $t {
                self.abs()
            }
        }
    };
}

macro_rules! make_l1norm_complex {
    ($t:ty) => {
        impl ArgminL1Norm<$t> for Complex<$t> {
            #[inline]
            fn l1_norm(&self) -> $t {
                self.l1_norm()
            }
        }
    };
}

macro_rules! make_l1norm_complex_unsigned {
    ($t:ty) => {
        impl ArgminL1Norm<$t> for Complex<$t> {
            #[inline]
            fn l1_norm(&self) -> $t {
                self.re + self.im
            }
        }
    };
}

make_l1norm!(isize);
make_l1norm!(i8);
make_l1norm!(i16);
make_l1norm!(i32);
make_l1norm!(i64);
make_l1norm_unsigned!(usize);
make_l1norm_unsigned!(u8);
make_l1norm_unsigned!(u16);
make_l1norm_unsigned!(u32);
make_l1norm_unsigned!(u64);
make_l1norm!(f32);
make_l1norm!(f64);
make_l1norm_complex!(isize);
make_l1norm_complex!(i8);
make_l1norm_complex!(i16);
make_l1norm_complex!(i32);
make_l1norm_complex!(i64);
make_l1norm_complex!(f32);
make_l1norm_complex!(f64);
make_l1norm_complex_unsigned!(usize);
make_l1norm_complex_unsigned!(u8);
make_l1norm_complex_unsigned!(u16);
make_l1norm_complex_unsigned!(u32);
make_l1norm_complex_unsigned!(u64);

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
                    let a = 8 as $t;
                    let res = <$t as ArgminL1Norm<$t>>::l1_norm(&a);
                    assert_relative_eq!(a as f64, res as f64, epsilon = std::f64::EPSILON);
                }
            }
        };
    }

    macro_rules! make_test_signed {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_norm_signed_ $t>]() {
                    let a = -8 as $t;
                    let res = <$t as ArgminL1Norm<$t>>::l1_norm(&a);
                    assert_relative_eq!(8 as f64, res as f64, epsilon = std::f64::EPSILON);
                }
            }
        };
    }

    macro_rules! make_test_complex_signed {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_norm_complex_signed_ $t>]() {
                    let a = Complex::new(-8 as $t, -4 as $t);
                    let res = <Complex<$t> as ArgminL1Norm<$t>>::l1_norm(&a);
                    assert_relative_eq!((8 as $t + 4 as $t) as f64, res as f64, epsilon = std::f64::EPSILON);
                }
            }
        };
    }

    macro_rules! make_test_complex {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_norm_complex_ $t>]() {
                    let a = Complex::new(8 as $t, 4 as $t);
                    let res = <Complex<$t> as ArgminL1Norm<$t>>::l1_norm(&a);
                    assert_relative_eq!((8 as $t + 4 as $t) as f64, res as f64, epsilon = std::f64::EPSILON);
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

    make_test_complex_signed!(isize);
    make_test_complex_signed!(i8);
    make_test_complex_signed!(i16);
    make_test_complex_signed!(i32);
    make_test_complex_signed!(i64);
    make_test_complex_signed!(f32);
    make_test_complex_signed!(f64);
    make_test_complex!(usize);
    make_test_complex!(u8);
    make_test_complex!(u16);
    make_test_complex!(u32);
    make_test_complex!(u64);
    make_test_complex!(isize);
    make_test_complex!(i8);
    make_test_complex!(i16);
    make_test_complex!(i32);
    make_test_complex!(i64);
    make_test_complex!(f32);
    make_test_complex!(f64);
}
