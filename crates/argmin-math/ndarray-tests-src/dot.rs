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
    use argmin_math::ArgminDot;
    use ndarray::array;
    use ndarray::{Array1, Array2};
    use num_complex::Complex;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_vec_vec_ $t>]() {
                    let a = array![1 as $t, 2 as $t, 3 as $t];
                    let b = array![4 as $t, 5 as $t, 6 as $t];
                    let res: $t = <Array1<$t> as ArgminDot<Array1<$t>, $t>>::dot(&a, &b);
                    assert_relative_eq!(res as f64, 32 as f64, epsilon = std::f64::EPSILON);
                }
            }

            item! {
                #[test]
                fn [<test_vec_scalar_ $t>]() {
                    let a = array![1 as $t, 2 as $t, 3 as $t];
                    let b = 2 as $t;
                    let product: Array1<$t> =
                        <Array1<$t> as ArgminDot<$t, Array1<$t>>>::dot(&a, &b);
                    let res = array![2 as $t, 4 as $t, 6 as $t];
                    for i in 0..3 {
                        assert_relative_eq!(res[i] as f64, product[i] as f64, epsilon = std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_scalar_vec_ $t>]() {
                    let a = array![1 as $t, 2 as $t, 3 as $t];
                    let b = 2 as $t;
                    let product: Array1<$t> =
                        <$t as ArgminDot<Array1<$t>, Array1<$t>>>::dot(&b, &a);
                    let res = array![2 as $t, 4 as $t, 6 as $t];
                    for i in 0..3 {
                        assert_relative_eq!(res[i] as f64, product[i] as f64, epsilon = std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mat_vec_ $t>]() {
                    let a = array![1 as $t, 2 as $t, 3 as $t];
                    let b = array![4 as $t, 5 as $t, 6 as $t];
                    let res = array![
                        [4 as $t, 5 as $t, 6 as $t],
                        [8 as $t, 10 as $t, 12 as $t],
                        [12 as $t, 15 as $t, 18 as $t]
                    ];
                    let product: Array2<$t> =
                        <Array1<$t> as ArgminDot<Array1<$t>, Array2<$t>>>::dot(&a, &b);
                    for i in 0..3 {
                        for j in 0..3 {
                            assert_relative_eq!(res[(i, j)] as f64, product[(i, j)] as f64, epsilon = std::f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mat_vec_2_ $t>]() {
                    let a = array![
                        [1 as $t, 2 as $t, 3 as $t],
                        [4 as $t, 5 as $t, 6 as $t],
                        [7 as $t, 8 as $t, 9 as $t]
                    ];
                    let b = array![1 as $t, 2 as $t, 3 as $t];
                    let res = array![14 as $t, 32 as $t, 50 as $t];
                    let product: Array1<$t> =
                        <Array2<$t> as ArgminDot<Array1<$t>, Array1<$t>>>::dot(&a, &b);
                    for i in 0..3 {
                        assert_relative_eq!(res[i] as f64, product[i] as f64, epsilon = std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mat_mat_ $t>]() {
                    let a = array![
                        [1 as $t, 2 as $t, 3 as $t],
                        [4 as $t, 5 as $t, 6 as $t],
                        [3 as $t, 2 as $t, 1 as $t]
                    ];
                    let b = array![
                        [3 as $t, 2 as $t, 1 as $t],
                        [6 as $t, 5 as $t, 4 as $t],
                        [2 as $t, 4 as $t, 3 as $t]
                    ];
                    let res = array![
                        [21 as $t, 24 as $t, 18 as $t],
                        [54 as $t, 57 as $t, 42 as $t],
                        [23 as $t, 20 as $t, 14 as $t]
                    ];
                    let product: Array2<$t> =
                        <Array2<$t> as ArgminDot<Array2<$t>, Array2<$t>>>::dot(&a, &b);
                    for i in 0..3 {
                        for j in 0..3 {
                            assert_relative_eq!(res[(i, j)] as f64, product[(i, j)] as f64, epsilon = std::f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mat_primitive_ $t>]() {
                    let a = array![
                        [1 as $t, 2 as $t, 3 as $t],
                        [4 as $t, 5 as $t, 6 as $t],
                        [3 as $t, 2 as $t, 1 as $t]
                    ];
                    let res = array![
                        [2 as $t, 4 as $t, 6 as $t],
                        [8 as $t, 10 as $t, 12 as $t],
                        [6 as $t, 4 as $t, 2 as $t]
                    ];
                    let product: Array2<$t> =
                        <Array2<$t> as ArgminDot<$t, Array2<$t>>>::dot(&a, &(2 as $t));
                    for i in 0..3 {
                        for j in 0..3 {
                            assert_relative_eq!(res[(i, j)] as f64, product[(i, j)] as f64, epsilon = std::f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_primitive_mat_ $t>]() {
                    let a = array![
                        [1 as $t, 2 as $t, 3 as $t],
                        [4 as $t, 5 as $t, 6 as $t],
                        [3 as $t, 2 as $t, 1 as $t]
                    ];
                    let res = array![
                        [2 as $t, 4 as $t, 6 as $t],
                        [8 as $t, 10 as $t, 12 as $t],
                        [6 as $t, 4 as $t, 2 as $t]
                    ];
                    let product: Array2<$t> =
                        <$t as ArgminDot<Array2<$t>, Array2<$t>>>::dot(&(2 as $t), &a);
                    for i in 0..3 {
                        for j in 0..3 {
                            assert_relative_eq!(res[(i, j)] as f64, product[(i, j)] as f64, epsilon = std::f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_vec_vec_complex_ $t>]() {
                    let a = array![
                        Complex::new(2 as $t, 2 as $t),
                        Complex::new(5 as $t, 2 as $t),
                        Complex::new(3 as $t, 2 as $t),
                    ];
                    let b = array![
                        Complex::new(5 as $t, 3 as $t),
                        Complex::new(2 as $t, 4 as $t),
                        Complex::new(8 as $t, 4 as $t),
                    ];
                    let res: Complex<$t> = <Array1<Complex<$t>> as ArgminDot<Array1<Complex<$t>>, Complex<$t>>>::dot(&a, &b);
                    let target = a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
                    assert_relative_eq!(res.re as f64, target.re as f64, epsilon = std::f64::EPSILON);
                    assert_relative_eq!(res.im as f64, target.im as f64, epsilon = std::f64::EPSILON);
                }
            }

            item! {
                #[test]
                fn [<test_vec_scalar_complex_ $t>]() {
                    let a = array![
                        Complex::new(2 as $t, 2 as $t),
                        Complex::new(5 as $t, 2 as $t),
                        Complex::new(3 as $t, 2 as $t),
                    ];
                    let b = Complex::new(4 as $t, 2 as $t);
                    let product: Array1<Complex<$t>> =
                        <Array1<Complex<$t>> as ArgminDot<Complex<$t>, Array1<Complex<$t>>>>::dot(&a, &b);
                    let res = array![a[0]*b, a[1]*b, a[2]*b];
                    for i in 0..3 {
                        assert_relative_eq!(res[i].re as f64, product[i].re as f64, epsilon = std::f64::EPSILON);
                        assert_relative_eq!(res[i].im as f64, product[i].im as f64, epsilon = std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_scalar_vec_complex_ $t>]() {
                    let a = array![
                        Complex::new(2 as $t, 2 as $t),
                        Complex::new(5 as $t, 2 as $t),
                        Complex::new(3 as $t, 2 as $t),
                    ];
                    let b = Complex::new(4 as $t, 2 as $t);
                    let product: Array1<Complex<$t>> =
                        <Complex<$t> as ArgminDot<Array1<Complex<$t>>, Array1<Complex<$t>>>>::dot(&b, &a);
                    let res = array![a[0]*b, a[1]*b, a[2]*b];
                    for i in 0..3 {
                        assert_relative_eq!(res[i].re as f64, product[i].re as f64, epsilon = std::f64::EPSILON);
                        assert_relative_eq!(res[i].im as f64, product[i].im as f64, epsilon = std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mat_vec_complex_ $t>]() {
                    let a = array![
                        Complex::new(2 as $t, 2 as $t),
                        Complex::new(5 as $t, 2 as $t),
                    ];
                    let b = array![
                        Complex::new(5 as $t, 1 as $t),
                        Complex::new(2 as $t, 1 as $t),
                    ];
                    let res = array![
                        [a[0]*b[0], a[0]*b[1]],
                        [a[1]*b[0], a[1]*b[1]],
                    ];
                    let product: Array2<Complex<$t>> =
                        <Array1<Complex<$t>> as ArgminDot<Array1<Complex<$t>>, Array2<Complex<$t>>>>::dot(&a, &b);
                    for i in 0..2 {
                        for j in 0..2 {
                            assert_relative_eq!(res[(i, j)].re as f64, product[(i, j)].re as f64, epsilon = std::f64::EPSILON);
                            assert_relative_eq!(res[(i, j)].im as f64, product[(i, j)].im as f64, epsilon = std::f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mat_vec_2_complex $t>]() {
                    let a = array![
                        [Complex::new(2 as $t, 2 as $t), Complex::new(5 as $t, 2 as $t)],
                        [Complex::new(2 as $t, 2 as $t), Complex::new(5 as $t, 2 as $t)],
                    ];
                    let b = array![
                        Complex::new(5 as $t, 1 as $t),
                        Complex::new(2 as $t, 1 as $t),
                    ];
                    let res = array![
                        a[(0, 0)] * b[0] + a[(0, 1)] * b[1],
                        a[(1, 0)] * b[0] + a[(1, 1)] * b[1],
                    ];
                    let product: Array1<Complex<$t>> =
                        <Array2<Complex<$t>> as ArgminDot<Array1<Complex<$t>>, Array1<Complex<$t>>>>::dot(&a, &b);
                    for i in 0..2 {
                        assert_relative_eq!(res[i].re as f64, product[i].re as f64, epsilon = std::f64::EPSILON);
                        assert_relative_eq!(res[i].im as f64, product[i].im as f64, epsilon = std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mat_mat_complex $t>]() {
                    let a = array![
                        [Complex::new(2 as $t, 1 as $t), Complex::new(5 as $t, 2 as $t)],
                        [Complex::new(4 as $t, 2 as $t), Complex::new(7 as $t, 1 as $t)],
                    ];
                    let b = array![
                        [Complex::new(2 as $t, 2 as $t), Complex::new(5 as $t, 1 as $t)],
                        [Complex::new(3 as $t, 1 as $t), Complex::new(4 as $t, 2 as $t)],
                    ];
                    let res = array![
                        [
                            a[(0, 0)] * b[(0, 0)] + a[(0, 1)] * b[(1, 0)],
                            a[(0, 0)] * b[(0, 1)] + a[(0, 1)] * b[(1, 1)]
                        ],
                        [
                            a[(1, 0)] * b[(0, 0)] + a[(1, 1)] * b[(1, 0)],
                            a[(1, 0)] * b[(0, 1)] + a[(1, 1)] * b[(1, 1)]
                        ],
                    ];
                    let product: Array2<Complex<$t>> =
                        <Array2<Complex<$t>> as ArgminDot<Array2<Complex<$t>>, Array2<Complex<$t>>>>::dot(&a, &b);
                    for i in 0..2 {
                        for j in 0..2 {
                            assert_relative_eq!(res[(i, j)].re as f64, product[(i, j)].re as f64, epsilon = std::f64::EPSILON);
                            assert_relative_eq!(res[(i, j)].im as f64, product[(i, j)].im as f64, epsilon = std::f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mat_primitive_complex_ $t>]() {
                    let a = array![
                        [Complex::new(2 as $t, 1 as $t), Complex::new(5 as $t, 2 as $t)],
                        [Complex::new(4 as $t, 2 as $t), Complex::new(7 as $t, 1 as $t)],
                    ];
                    let b = Complex::new(4 as $t, 1 as $t);
                    let res = array![
                        [a[(0, 0)] * b, a[(0, 1)] * b],
                        [a[(1, 0)] * b, a[(1, 1)] * b]
                    ];
                    let product: Array2<Complex<$t>> =
                        <Array2<Complex<$t>> as ArgminDot<Complex<$t>, Array2<Complex<$t>>>>::dot(&a, &b);
                    for i in 0..2 {
                        for j in 0..2 {
                            assert_relative_eq!(res[(i, j)].re as f64, product[(i, j)].re as f64, epsilon = std::f64::EPSILON);
                            assert_relative_eq!(res[(i, j)].im as f64, product[(i, j)].im as f64, epsilon = std::f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_primitive_mat_complex_ $t>]() {
                    let a = array![
                        [Complex::new(2 as $t, 1 as $t), Complex::new(5 as $t, 2 as $t)],
                        [Complex::new(4 as $t, 2 as $t), Complex::new(7 as $t, 1 as $t)],
                    ];
                    let b = Complex::new(4 as $t, 1 as $t);
                    let res = array![
                        [a[(0, 0)] * b, a[(0, 1)] * b],
                        [a[(1, 0)] * b, a[(1, 1)] * b],
                    ];
                    let product: Array2<Complex<$t>> =
                        <Complex<$t> as ArgminDot<Array2<Complex<$t>>, Array2<Complex<$t>>>>::dot(&b, &a);
                    for i in 0..2 {
                        for j in 0..2 {
                            assert_relative_eq!(res[(i, j)].re as f64, product[(i, j)].re as f64, epsilon = std::f64::EPSILON);
                            assert_relative_eq!(res[(i, j)].im as f64, product[(i, j)].im as f64, epsilon = std::f64::EPSILON);
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
    make_test!(f32);
    make_test!(f64);
}
