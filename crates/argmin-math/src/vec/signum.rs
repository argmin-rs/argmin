// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminSignum;
use num_complex::Complex;

macro_rules! make_signum {
    ($t:ty) => {
        impl ArgminSignum for Vec<$t> {
            fn signum(mut self) -> Self {
                for x in &mut self {
                    *x = x.signum();
                }
                self
            }
        }
    };
}

macro_rules! make_signum_complex {
    ($t:ty) => {
        impl ArgminSignum for Vec<$t> {
            fn signum(mut self) -> Self {
                for x in &mut self {
                    x.re = x.re.signum();
                    x.im = x.im.signum();
                }
                self
            }
        }
    };
}

make_signum!(i8);
make_signum!(i16);
make_signum!(i32);
make_signum!(i64);
make_signum!(f32);
make_signum!(f64);
make_signum_complex!(Complex<i8>);
make_signum_complex!(Complex<i16>);
make_signum_complex!(Complex<i32>);
make_signum_complex!(Complex<i64>);
make_signum_complex!(Complex<f32>);
make_signum_complex!(Complex<f64>);

#[cfg(test)]
mod tests {
    use super::*;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_signum_complex_ $t>]() {
                    let x = vec![
                        Complex::new(1 as $t, 2 as $t),
                        Complex::new(4 as $t, -3 as $t),
                        Complex::new(-8 as $t, 4 as $t),
                        Complex::new(-8 as $t, -1 as $t),
                    ];
                    let y = vec![
                        Complex::new(1 as $t, 1 as $t),
                        Complex::new(1 as $t, -1 as $t),
                        Complex::new(-1 as $t, 1 as $t),
                        Complex::new(-1 as $t, -1 as $t),
                    ];
                    let res = <Vec<Complex<$t>> as ArgminSignum>::signum(x);
                    for i in 0..4 {
                        let tmp = y[i] - res[i];
                        let norm = ((tmp.re * tmp.re + tmp.im * tmp.im) as f64).sqrt();
                        assert!(norm < f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_signum_ $t>]() {
                    let x = vec![1 as $t, -4 as $t, 8 as $t];
                    let y = vec![1 as $t, -1 as $t, 1 as $t];
                    let res = <Vec<$t> as ArgminSignum>::signum(x);
                    for i in 0..3 {
                        let diff = (y[i] - res[i]).abs() as f64;
                        assert!(diff < f64::EPSILON);
                    }
                }
            }
        };
    }

    make_test!(i8);
    make_test!(i16);
    make_test!(i32);
    make_test!(i64);
    make_test!(f32);
    make_test!(f64);
}
