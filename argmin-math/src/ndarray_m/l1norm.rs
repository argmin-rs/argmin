// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminL1Norm;
use ndarray::Array1;
use num_complex::Complex;

macro_rules! make_l1norm_float {
    ($t:ty) => {
        impl ArgminL1Norm<$t> for Array1<$t> {
            #[inline]
            fn l1_norm(&self) -> $t {
                self.iter().map(|a| a.abs()).sum()
            }
        }
    };
}

macro_rules! make_l1norm_complex_float {
    ($t:ty) => {
        impl ArgminL1Norm<Complex<$t>> for Array1<Complex<$t>> {
            #[inline]
            fn l1_norm(&self) -> Complex<$t> {
                self.iter().map(|a| a.l1_norm()).sum::<$t>().into()
            }
        }

        impl ArgminL1Norm<$t> for Array1<Complex<$t>> {
            #[inline]
            fn l1_norm(&self) -> $t {
                self.iter().map(|a| a.l1_norm()).sum()
            }
        }
    };
}

macro_rules! make_l1norm_unsigned {
    ($t:ty) => {
        impl ArgminL1Norm<$t> for Array1<$t> {
            #[inline]
            fn l1_norm(&self) -> $t {
                self.sum()
            }
        }
    };
}

macro_rules! make_l1norm_integer {
    ($t:ty) => {
        impl ArgminL1Norm<$t> for Array1<$t> {
            #[inline]
            fn l1_norm(&self) -> $t {
                self.iter().map(|a| a.abs()).sum()
            }
        }
    };
}

make_l1norm_integer!(isize);
make_l1norm_unsigned!(usize);
make_l1norm_integer!(i8);
make_l1norm_integer!(i16);
make_l1norm_integer!(i32);
make_l1norm_integer!(i64);
make_l1norm_unsigned!(u8);
make_l1norm_unsigned!(u16);
make_l1norm_unsigned!(u32);
make_l1norm_unsigned!(u64);
make_l1norm_float!(f32);
make_l1norm_float!(f64);
make_l1norm_complex_float!(f32);
make_l1norm_complex_float!(f64);

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
                    let res = <Array1<$t> as ArgminL1Norm<$t>>::l1_norm(&a);
                    let target = 7 as $t;
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
                    let res = <Array1<$t> as ArgminL1Norm<$t>>::l1_norm(&a);
                    let target = 7 as $t;
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
