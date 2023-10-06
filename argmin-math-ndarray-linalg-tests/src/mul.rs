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
    use argmin_math::ArgminMul;
    use ndarray::{Array1, Array2};
    use num_complex::Complex;
       use ndarray::array;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_mul_vec_scalar_ $t>]() {
                    let a = array![1 as $t, 4 as $t, 8 as $t];
                    let b = 2 as $t;
                    let target = array![2 as $t, 8 as $t, 16 as $t];
                    let res = <Array1<$t> as ArgminMul<$t, Array1<$t>>>::mul(&a, &b);
                    for i in 0..3 {
                        assert!(((target[i] - res[i]) as f64).abs() < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mul_vec_scalar_complex_ $t>]() {
                    let a = array![
                        Complex::new(5 as $t, 3 as $t),
                        Complex::new(8 as $t, 2 as $t)
                    ];
                    let b = Complex::new(2 as $t, 3 as $t);
                    let target = array![a[0] * b, a[1] * b];
                    let res = <Array1<Complex<$t>> as ArgminMul<Complex<$t>, Array1<Complex<$t>>>>::mul(&a, &b);
                    for i in 0..2 {
                        assert!((target[i].re as f64 - res[i].re as f64).abs() < std::f64::EPSILON);
                        assert!((target[i].im as f64 - res[i].im as f64).abs() < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mul_vec_scalar_complex_2_ $t>]() {
                    let a = array![
                        Complex::new(5 as $t, 3 as $t),
                        Complex::new(8 as $t, 2 as $t)
                    ];
                    let b = 2 as $t;
                    let target = array![a[0] * b, a[1] * b];
                    let res = <Array1<Complex<$t>> as ArgminMul<$t, Array1<Complex<$t>>>>::mul(&a, &b);
                    for i in 0..2 {
                        assert!((target[i].re as f64 - res[i].re as f64).abs() < std::f64::EPSILON);
                        assert!((target[i].im as f64 - res[i].im as f64).abs() < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mul_scalar_vec_ $t>]() {
                    let a = array![1 as $t, 4 as $t, 8 as $t];
                    let b = 2 as $t;
                    let target = array![2 as $t, 8 as $t, 16 as $t];
                    let res = <$t as ArgminMul<Array1<$t>, Array1<$t>>>::mul(&b, &a);
                    for i in 0..3 {
                        assert!(((target[i] - res[i]) as f64).abs() < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mul_scalar_vec_complex_ $t>]() {
                    let a = array![
                        Complex::new(5 as $t, 3 as $t),
                        Complex::new(8 as $t, 2 as $t)
                    ];
                    let b = Complex::new(2 as $t, 3 as $t);
                    let target = array![a[0] * b, a[1] * b];
                    let res = <Complex<$t> as ArgminMul<Array1<Complex<$t>>, Array1<Complex<$t>>>>::mul(&b, &a);
                    for i in 0..2 {
                        assert!((target[i].re as f64 - res[i].re as f64).abs() < std::f64::EPSILON);
                        assert!((target[i].im as f64 - res[i].im as f64).abs() < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mul_scalar_vec_complex_2_ $t>]() {
                    let a = array![
                        Complex::new(5 as $t, 3 as $t),
                        Complex::new(8 as $t, 2 as $t)
                    ];
                    let b = 2 as $t;
                    let target = array![a[0] * b, a[1] * b];
                    let res = <$t as ArgminMul<Array1<Complex<$t>>, Array1<Complex<$t>>>>::mul(&b, &a);
                    for i in 0..2 {
                        assert!((target[i].re as f64 - res[i].re as f64).abs() < std::f64::EPSILON);
                        assert!((target[i].im as f64 - res[i].im as f64).abs() < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mul_vec_vec_ $t>]() {
                    let a = array![1 as $t, 4 as $t, 8 as $t];
                    let b = array![2 as $t, 3 as $t, 4 as $t];
                    let target = array![2 as $t, 12 as $t, 32 as $t];
                    let res = <Array1<$t> as ArgminMul<Array1<$t>, Array1<$t>>>::mul(&a, &b);
                    for i in 0..3 {
                        assert!(((target[i] - res[i]) as f64).abs() < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mul_vec_vec_complex_ $t>]() {
                    let a = array![
                        Complex::new(5 as $t, 3 as $t),
                        Complex::new(8 as $t, 2 as $t)
                    ];
                    let b = array![
                        Complex::new(2 as $t, 3 as $t),
                        Complex::new(1 as $t, 2 as $t)
                    ];
                    let target = array![a[0]*b[0], a[1]*b[1]];
                    let res = <Array1<Complex<$t>> as ArgminMul<Array1<Complex<$t>>, Array1<Complex<$t>>>>::mul(&a, &b);
                    for i in 0..2 {
                        assert!((target[i].re as f64 - res[i].re as f64).abs() < std::f64::EPSILON);
                        assert!((target[i].im as f64 - res[i].im as f64).abs() < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_mul_vec_vec_panic_ $t>]() {
                    let a = array![1 as $t, 4 as $t];
                    let b = array![41 as $t, 38 as $t, 34 as $t];
                    <Array1<$t> as ArgminMul<Array1<$t>, Array1<$t>>>::mul(&a, &b);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_mul_vec_vec_panic_2_ $t>]() {
                    let a = array![];
                    let b = array![41 as $t, 38 as $t, 34 as $t];
                    <Array1<$t> as ArgminMul<Array1<$t>, Array1<$t>>>::mul(&a, &b);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_mul_vec_vec_panic_3_ $t>]() {
                    let a = array![41 as $t, 38 as $t, 34 as $t];
                    let b = array![];
                    <Array1<$t> as ArgminMul<Array1<$t>, Array1<$t>>>::mul(&a, &b);
                }
            }

            item! {
                #[test]
                fn [<test_mul_mat_mat_ $t>]() {
                    let a = array![
                        [1 as $t, 4 as $t, 8 as $t],
                        [2 as $t, 5 as $t, 9 as $t]
                    ];
                    let b = array![
                        [2 as $t, 3 as $t, 4 as $t],
                        [3 as $t, 4 as $t, 5 as $t]
                    ];
                    let target = array![
                        [2 as $t, 12 as $t, 32 as $t],
                        [6 as $t, 20 as $t, 45 as $t]
                    ];
                    let res = <Array2<$t> as ArgminMul<Array2<$t>, Array2<$t>>>::mul(&a, &b);
                    for i in 0..3 {
                        for j in 0..2 {
                        assert!(((target[(j, i)] - res[(j, i)]) as f64).abs() < std::f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mul_mat_mat_complex_ $t>]() {
                    let a = array![
                        [Complex::new(5 as $t, 3 as $t), Complex::new(8 as $t, 2 as $t)],
                        [Complex::new(4 as $t, 2 as $t), Complex::new(7 as $t, 1 as $t)],
                        [Complex::new(3 as $t, 1 as $t), Complex::new(6 as $t, 2 as $t)],
                    ];
                    let b = array![
                        [Complex::new(5 as $t, 3 as $t), Complex::new(8 as $t, 2 as $t)],
                        [Complex::new(4 as $t, 2 as $t), Complex::new(7 as $t, 1 as $t)],
                        [Complex::new(3 as $t, 1 as $t), Complex::new(6 as $t, 2 as $t)],
                    ];
                    let target = array![
                        [a[(0, 0)] * b[(0, 0)], a[(0, 1)] * b[(0, 1)]],
                        [a[(1, 0)] * b[(1, 0)], a[(1, 1)] * b[(1, 1)]],
                        [a[(2, 0)] * b[(2, 0)], a[(2, 1)] * b[(2, 1)]],
                    ];
                    let res = <Array2<Complex<$t>> as ArgminMul<Array2<Complex<$t>>, Array2<Complex<$t>>>>::mul(&a, &b);
                    for i in 0..2 {
                        for j in 0..3 {
                            assert!((target[(j, i)].re as f64 - res[(j, i)].re as f64).abs() < std::f64::EPSILON);
                            assert!((target[(j, i)].im as f64 - res[(j, i)].im as f64).abs() < std::f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_mul_mat_mat_panic_2_ $t>]() {
                    let a = array![
                        [1 as $t, 4 as $t, 8 as $t],
                        [2 as $t, 5 as $t, 9 as $t]
                    ];
                    let b = array![
                        [41 as $t, 38 as $t],
                    ];
                    <Array2<$t> as ArgminMul<Array2<$t>, Array2<$t>>>::mul(&a, &b);
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_mul_mat_mat_panic_3_ $t>]() {
                    let a = array![
                        [1 as $t, 4 as $t, 8 as $t],
                        [2 as $t, 5 as $t, 9 as $t]
                    ];
                    let b = array![[]];
                    <Array2<$t> as ArgminMul<Array2<$t>, Array2<$t>>>::mul(&a, &b);
                }
            }

            item! {
                #[test]
                fn [<test_mul_scalar_mat_1_ $t>]() {
                    let a = array![
                        [1 as $t, 4 as $t, 8 as $t],
                        [2 as $t, 5 as $t, 9 as $t]
                    ];
                    let b = 2 as $t;
                    let target = array![
                        [2 as $t, 8 as $t, 16 as $t],
                        [4 as $t, 10 as $t, 18 as $t]
                    ];
                    let res = <Array2<$t> as ArgminMul<$t, Array2<$t>>>::mul(&a, &b);
                    for i in 0..3 {
                        for j in 0..2 {
                        assert!(((target[(j, i)] - res[(j, i)]) as f64).abs() < std::f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mul_scalar_mat_1_complex_ $t>]() {
                    let a = array![
                        [Complex::new(5 as $t, 3 as $t), Complex::new(8 as $t, 2 as $t)],
                        [Complex::new(4 as $t, 2 as $t), Complex::new(7 as $t, 1 as $t)],
                        [Complex::new(3 as $t, 1 as $t), Complex::new(6 as $t, 2 as $t)],
                    ];
                    let b = Complex::new(3 as $t, 2 as $t);
                    let target = array![
                        [a[(0, 0)] * b, a[(0, 1)] * b],
                        [a[(1, 0)] * b, a[(1, 1)] * b],
                        [a[(2, 0)] * b, a[(2, 1)] * b],
                    ];
                    let res = <Array2<Complex<$t>> as ArgminMul<Complex<$t>, Array2<Complex<$t>>>>::mul(&a, &b);
                    for i in 0..2 {
                        for j in 0..3 {
                            assert!((target[(j, i)].re as f64 - res[(j, i)].re as f64).abs() < std::f64::EPSILON);
                            assert!((target[(j, i)].im as f64 - res[(j, i)].im as f64).abs() < std::f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mul_scalar_mat_1_complex_2_ $t>]() {
                    let a = array![
                        [Complex::new(5 as $t, 3 as $t), Complex::new(8 as $t, 2 as $t)],
                        [Complex::new(4 as $t, 2 as $t), Complex::new(7 as $t, 1 as $t)],
                        [Complex::new(3 as $t, 1 as $t), Complex::new(6 as $t, 2 as $t)],
                    ];
                    let b = 3 as $t;
                    let target = array![
                        [a[(0, 0)] * b, a[(0, 1)] * b],
                        [a[(1, 0)] * b, a[(1, 1)] * b],
                        [a[(2, 0)] * b, a[(2, 1)] * b],
                    ];
                    let res = <Array2<Complex<$t>> as ArgminMul<$t, Array2<Complex<$t>>>>::mul(&a, &b);
                    for i in 0..2 {
                        for j in 0..3 {
                            assert!((target[(j, i)].re as f64 - res[(j, i)].re as f64).abs() < std::f64::EPSILON);
                            assert!((target[(j, i)].im as f64 - res[(j, i)].im as f64).abs() < std::f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mul_scalar_mat_2_ $t>]() {
                    let b = array![
                        [1 as $t, 4 as $t, 8 as $t],
                        [2 as $t, 5 as $t, 9 as $t]
                    ];
                    let a = 2 as $t;
                    let target = array![
                        [2 as $t, 8 as $t, 16 as $t],
                        [4 as $t, 10 as $t, 18 as $t]
                    ];
                    let res = <$t as ArgminMul<Array2<$t>, Array2<$t>>>::mul(&a, &b);
                    for i in 0..3 {
                        for j in 0..2 {
                        assert!(((target[(j, i)] - res[(j, i)]) as f64).abs() < std::f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mul_scalar_mat_2_complex_ $t>]() {
                    let a = array![
                        [Complex::new(5 as $t, 3 as $t), Complex::new(8 as $t, 2 as $t)],
                        [Complex::new(4 as $t, 2 as $t), Complex::new(7 as $t, 1 as $t)],
                        [Complex::new(3 as $t, 1 as $t), Complex::new(6 as $t, 2 as $t)],
                    ];
                    let b = Complex::new(3 as $t, 2 as $t);
                    let target = array![
                        [a[(0, 0)] * b, a[(0, 1)] * b],
                        [a[(1, 0)] * b, a[(1, 1)] * b],
                        [a[(2, 0)] * b, a[(2, 1)] * b],
                    ];
                    let res = <Complex<$t> as ArgminMul<Array2<Complex<$t>>, Array2<Complex<$t>>>>::mul(&b, &a);
                    for i in 0..2 {
                        for j in 0..3 {
                            assert!((target[(j, i)].re as f64 - res[(j, i)].re as f64).abs() < std::f64::EPSILON);
                            assert!((target[(j, i)].im as f64 - res[(j, i)].im as f64).abs() < std::f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mul_scalar_mat_2_complex_2_ $t>]() {
                    let a = array![
                        [Complex::new(5 as $t, 3 as $t), Complex::new(8 as $t, 2 as $t)],
                        [Complex::new(4 as $t, 2 as $t), Complex::new(7 as $t, 1 as $t)],
                        [Complex::new(3 as $t, 1 as $t), Complex::new(6 as $t, 2 as $t)],
                    ];
                    let b = 3 as $t;
                    let target = array![
                        [a[(0, 0)] * b, a[(0, 1)] * b],
                        [a[(1, 0)] * b, a[(1, 1)] * b],
                        [a[(2, 0)] * b, a[(2, 1)] * b],
                    ];
                    let res = <$t as ArgminMul<Array2<Complex<$t>>, Array2<Complex<$t>>>>::mul(&b, &a);
                    for i in 0..2 {
                        for j in 0..3 {
                            assert!((target[(j, i)].re as f64 - res[(j, i)].re as f64).abs() < std::f64::EPSILON);
                            assert!((target[(j, i)].im as f64 - res[(j, i)].im as f64).abs() < std::f64::EPSILON);
                        }
                    }
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
    make_test!(isize);
    make_test!(usize);
    make_test!(f32);
    make_test!(f64);
}
