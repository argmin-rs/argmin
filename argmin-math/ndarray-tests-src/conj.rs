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
    use argmin_math::ArgminConj;
    use ndarray::{Array1, Array2};
    use num_complex::Complex;
    use ndarray::array;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_conj_complex_ $t>]() {
                    let a = Array1::from(vec![
                        Complex::new(1 as $t, 2 as $t),
                        Complex::new(4 as $t, -3 as $t),
                        Complex::new(8 as $t, 0 as $t)
                    ]);
                    let b = vec![
                        Complex::new(1 as $t, -2 as $t),
                        Complex::new(4 as $t, 3 as $t),
                        Complex::new(8 as $t, 0 as $t)
                    ];
                    let res = <Array1<Complex<$t>> as ArgminConj>::conj(&a);
                    for i in 0..3 {
                        let tmp = b[i] - res[i];
                        let norm = ((tmp.re * tmp.re + tmp.im * tmp.im) as f64).sqrt();
                        assert!(norm  < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_conj_complex_mat_ $t>]() {
                    let a = array![
                        [Complex::new(1 as $t, 2 as $t), Complex::new(4 as $t, -5 as $t)],
                        [Complex::new(3 as $t, -5 as $t), Complex::new(8 as $t, 3 as $t)],
                        [Complex::new(4 as $t, -2 as $t), Complex::new(9 as $t, -9 as $t)],
                    ];
                    let b = array![
                        [Complex::new(1 as $t, -2 as $t), Complex::new(4 as $t, 5 as $t)],
                        [Complex::new(3 as $t, 5 as $t), Complex::new(8 as $t, -3 as $t)],
                        [Complex::new(4 as $t, 2 as $t), Complex::new(9 as $t, 9 as $t)],
                    ];
                    let res = <Array2<Complex<$t>> as ArgminConj>::conj(&a);
                    for i in 0..2 {
                        for j in 0..3 {
                            assert!((b[(j, i)].re as f64 - res[(j, i)].re as f64).abs() < std::f64::EPSILON);
                            assert!((b[(j, i)].im as f64 - res[(j, i)].im as f64).abs() < std::f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_conj_ $t>]() {
                    let a = Array1::from(vec![1 as $t, 4 as $t, 8 as $t]);
                    let res = <Array1<$t> as ArgminConj>::conj(&a);
                    for i in 0..3 {
                        assert!((a[i] as f64 - res[i] as f64).abs() < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_conj_mat_ $t>]() {
                    let a = array![
                        [1 as $t, 4 as $t, 8 as $t],
                        [2 as $t, 5 as $t, 9 as $t],
                    ];
                    let res = <Array2<$t> as ArgminConj>::conj(&a);
                    for i in 0..3 {
                        for j in 0..2 {
                            assert!((a[(j, i)] as f64 - res[(j, i)] as f64).abs() < std::f64::EPSILON);
                        }
                    }
                }
            }
        };
    }

    make_test!(isize);
    make_test!(i8);
    make_test!(i16);
    make_test!(i32);
    make_test!(i64);
    make_test!(f32);
    make_test!(f64);

}