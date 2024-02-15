// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Ackley test function
//!
//! Defined as
//!
//! `f(x_1, x_2, ..., x_n) = - a * exp( -b \sqrt{\frac{1}{d}\sum_{i=1}^n x_i^2 ) -
//! exp( \frac{1}{d} cos(c * x_i) ) + a + exp(1)`
//!
//! where `x_i \in [-32.768, 32.768]` and usually `a = 10`, `b = 0.2` and `c = 2*pi`
//!
//! The global minimum is at `f(x_1, x_2, ..., x_n) = f(0, 0, ..., 0) = 0`.

use num::{Float, FromPrimitive};
use std::f64::consts::PI;
use std::iter::Sum;

/// Ackley test function
///
/// Defined as
///
/// `f(x_1, x_2, ..., x_n) = - a * exp( -b \sqrt{\frac{1}{d}\sum_{i=1}^n x_i^2 ) -
/// exp( \frac{1}{d} cos(c * x_i) ) + a + exp(1)`
///
/// where `x_i \in [-32.768, 32.768]` and usually `a = 10`, `b = 0.2` and `c = 2*pi`
///
/// The global minimum is at `f(x_1, x_2, ..., x_n) = f(0, 0, ..., 0) = 0`.
pub fn ackley<T>(param: &[T]) -> T
where
    T: Float + FromPrimitive + Sum,
{
    ackley_abc(
        param,
        T::from_f64(20.0).unwrap(),
        T::from_f64(0.2).unwrap(),
        T::from_f64(2.0 * PI).unwrap(),
    )
}

/// Ackley test function
///
/// The same as `ackley`; however, it allows to set the parameters a, b and c.
pub fn ackley_abc<T>(param: &[T], a: T, b: T, c: T) -> T
where
    T: Float + FromPrimitive + Sum,
{
    let num1 = T::from_f64(1.0).unwrap();
    let n = T::from_usize(param.len()).unwrap();
    -a * (-b * ((num1 / n) * param.iter().map(|x| x.powi(2)).sum()).sqrt()).exp()
        - ((num1 / n) * param.iter().map(|x| (c * *x).cos()).sum()).exp()
        + a
        + num1.exp()
}

/// Derivative of Ackley test function
///
/// Calls `ackley_abc_derivative` with `a = 10`, `b = 0.2` and `c = 2*pi`
pub fn ackley_derivative<T>(param: &[T]) -> Vec<T>
where
    T: Float + FromPrimitive + Sum,
{
    ackley_abc_derivative(
        param,
        T::from_f64(20.0).unwrap(),
        T::from_f64(0.2).unwrap(),
        T::from_f64(2.0 * PI).unwrap(),
    )
}

/// Derivative of Ackley test function
///
/// The same as `ackley_derivative`; however, it allows to set the parameters a, b and c.
pub fn ackley_abc_derivative<T>(param: &[T], a: T, b: T, c: T) -> Vec<T>
where
    T: Float + FromPrimitive + Sum,
{
    let d = T::from_usize(param.len()).unwrap();
    let n0 = T::from_f64(0.0).unwrap();
    let eps = T::epsilon();

    let norm = param.iter().map(|x| x.powi(2)).sum::<T>().sqrt();

    let f1 = (c * (param.iter().map(|&x| (c * x).cos()).sum::<T>() / d).exp()) / d;
    let f2 = if norm <= eps {
        n0
    } else {
        (a * b * (-b * norm / d.sqrt()).exp()) / (d.sqrt() * norm)
    };

    param
        .iter()
        .map(|&x| ((c * x).sin() * f1) + x * f2)
        .collect()
}

/// Derivative of Ackley test function
///
/// Calls `ackley_abc_derivative_const` with `a = 10`, `b = 0.2` and `c = 2*pi`
///
/// This is the const generics version, which requires the number of parameters to be known
/// at compile time.
pub fn ackley_derivative_const<const N: usize, T>(param: &[T; N]) -> [T; N]
where
    T: Float + FromPrimitive + Sum,
{
    ackley_abc_derivative_const(
        param,
        T::from_f64(20.0).unwrap(),
        T::from_f64(0.2).unwrap(),
        T::from_f64(2.0 * PI).unwrap(),
    )
}

/// Derivative of Ackley test function
///
/// The same as `ackley_derivative`; however, it allows to set the parameters a, b and c.
///
/// This is the const generics version, which requires the number of parameters to be known
/// at compile time.
pub fn ackley_abc_derivative_const<const N: usize, T>(param: &[T; N], a: T, b: T, c: T) -> [T; N]
where
    T: Float + FromPrimitive + Sum,
{
    let d = T::from_usize(param.len()).unwrap();
    let n0 = T::from_f64(0.0).unwrap();
    let eps = T::epsilon();

    let norm = param.iter().map(|x| x.powi(2)).sum::<T>().sqrt();

    let f1 = (c * (param.iter().map(|&x| (c * x).cos()).sum::<T>() / d).exp()) / d;
    let f2 = if norm <= eps {
        n0
    } else {
        (a * b * (-b * norm / d.sqrt()).exp()) / (d.sqrt() * norm)
    };

    let mut out = [n0; N];
    param
        .iter()
        .zip(out.iter_mut())
        .map(|(&x, o)| *o = ((c * x).sin() * f1) + x * f2)
        .count();

    out
}

/// Hessian of Ackley test function
///
/// Calls `ackley_abc_hessian` with `a = 10`, `b = 0.2` and `c = 2*pi`
pub fn ackley_hessian<T>(param: &[T]) -> Vec<Vec<T>>
where
    T: Float + FromPrimitive + Sum + std::fmt::Debug,
{
    ackley_abc_hessian(
        param,
        T::from_f64(20.0).unwrap(),
        T::from_f64(0.2).unwrap(),
        T::from_f64(2.0 * PI).unwrap(),
    )
}

/// Hessian of Ackley test function
///
/// The same as `ackley_hessian`; however, it allows to set the parameters a, b and c.
pub fn ackley_abc_hessian<T>(param: &[T], a: T, b: T, c: T) -> Vec<Vec<T>>
where
    T: Float + FromPrimitive + Sum,
{
    let du = param.len();
    assert!(du >= 1);
    let d = T::from_usize(du).unwrap();
    let n0 = T::from_f64(0.0).unwrap();
    let eps = T::epsilon();

    // rename for convenience
    let x = param;

    let sqsum = param.iter().map(|x| x.powi(2)).sum::<T>();
    let norm = sqsum.sqrt();
    let nexp = (-b * norm / d.sqrt()).exp();

    let f1 = -c.powi(2) * (x.iter().map(|&x| (c * x).cos()).sum::<T>() / d).exp();
    let f2 = (a * b.powi(2) * nexp) / (d * sqsum);
    let f3 = (a * b * nexp) / (d.sqrt() * norm.powi(3));
    let f4 = (a * b * nexp) / (d.sqrt() * norm);
    let f5 = (a * b.powi(2) * nexp) / (d * sqsum);
    let f6 = (a * b * nexp) / (d.sqrt() * norm.powi(3));

    let mut out = vec![vec![n0; du]; du];
    for i in 0..du {
        for j in 0..du {
            if i == j {
                out[i][j] = (c * x[i]).sin().powi(2) * f1 / d.powi(2)
                    - (c * x[i]).cos() * f1 / d
                    - if norm <= eps {
                        n0
                    } else {
                        x[i].powi(2) * (f2 + f3) - f4
                    };
            } else {
                let tmp = (c * x[i]).sin() * (c * x[j]).sin() * f1 / d.powi(2)
                    + x[i] * x[j] * if norm <= eps { n0 } else { -f5 - f6 };
                out[i][j] = tmp;
                out[j][i] = tmp;
            };
        }
    }

    out
}

/// Hessian of Ackley test function
///
/// Calls `ackley_abc_hessian` with `a = 10`, `b = 0.2` and `c = 2*pi`
///
/// This is the const generics version, which requires the number of parameters to be known
/// at compile time.
pub fn ackley_hessian_const<const N: usize, T>(param: &[T; N]) -> [[T; N]; N]
where
    T: Float + FromPrimitive + Sum,
{
    ackley_abc_hessian_const(
        param,
        T::from_f64(20.0).unwrap(),
        T::from_f64(0.2).unwrap(),
        T::from_f64(2.0 * PI).unwrap(),
    )
}

/// Hessian of Ackley test function
///
/// The same as `ackley_hessian`; however, it allows to set the parameters a, b and c.
pub fn ackley_abc_hessian_const<const N: usize, T>(param: &[T; N], a: T, b: T, c: T) -> [[T; N]; N]
where
    T: Float + FromPrimitive + Sum,
{
    assert!(N >= 1);
    let d = T::from_usize(N).unwrap();
    let n0 = T::from_f64(0.0).unwrap();
    let eps = T::epsilon();

    // rename for convenience
    let x = param;

    let sqsum = param.iter().map(|x| x.powi(2)).sum::<T>();
    let norm = sqsum.sqrt();
    let nexp = (-b * norm / d.sqrt()).exp();

    let f1 = -c.powi(2) * (x.iter().map(|&x| (c * x).cos()).sum::<T>() / d).exp();
    let f2 = (a * b.powi(2) * nexp) / (d * sqsum);
    let f3 = (a * b * nexp) / (d.sqrt() * norm.powi(3));
    let f4 = (a * b * nexp) / (d.sqrt() * norm);
    let f5 = (a * b.powi(2) * nexp) / (d * sqsum);
    let f6 = (a * b * nexp) / (d.sqrt() * norm.powi(3));

    let mut out = [[n0; N]; N];
    for i in 0..N {
        for j in 0..N {
            if i == j {
                out[i][j] = (c * x[i]).sin().powi(2) * f1 / d.powi(2)
                    - (c * x[i]).cos() * f1 / d
                    - if norm <= eps {
                        n0
                    } else {
                        x[i].powi(2) * (f2 + f3) - f4
                    };
            } else {
                let tmp = (c * x[i]).sin() * (c * x[j]).sin() * f1 / d.powi(2)
                    + x[i] * x[j] * if norm <= eps { n0 } else { -f5 - f6 };
                out[i][j] = tmp;
                out[j][i] = tmp;
            };
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use finitediff::FiniteDiff;
    use proptest::prelude::*;
    use std::{f32, f64};

    #[test]
    fn test_ackley_optimum() {
        assert_relative_eq!(
            ackley(&[0.0_f32, 0.0_f32, 0.0_f32]),
            0.0,
            epsilon = f32::EPSILON * 10_f32
        );
        assert_relative_eq!(
            ackley(&[0.0_f64, 0.0_f64, 0.0_f64]),
            0.0,
            epsilon = f64::EPSILON * 3_f64
        );

        let deriv = ackley_derivative(&[0.0_f64, 0.0_f64, 0.0_f64]);
        for i in 0..3 {
            assert_relative_eq!(deriv[i], 0.0, epsilon = f64::EPSILON * 3_f64);
        }

        let deriv = ackley_derivative_const(&[0.0_f64, 0.0_f64, 0.0_f64]);
        for i in 0..3 {
            assert_relative_eq!(deriv[i], 0.0, epsilon = f64::EPSILON * 3_f64);
        }
    }

    proptest! {
        #[test]
        fn test_parameters(a in -5.0..5.0, b in -5.0..5.0, c in -5.0..5.0) {
            let param = [a, b, c];
            assert_relative_eq!(
                ackley(&param),
                ackley_abc(&param, 20.0, 0.2, 2.0 * PI),
                epsilon = f64::EPSILON
            );

            let deriv1 = ackley_derivative(&param);
            let deriv2 = ackley_abc_derivative(&param, 20.0, 0.2, 2.0 * PI);
            for i in 0..3 {
                assert_relative_eq!(deriv1[i], deriv2[i], epsilon = f64::EPSILON);
            }

            let deriv1 = ackley_derivative_const(&param);
            let deriv2 = ackley_abc_derivative_const(&param, 20.0, 0.2, 2.0 * PI);
            for i in 0..3 {
                assert_relative_eq!(deriv1[i], deriv2[i], epsilon = f64::EPSILON);
            }

            let hessian1 = ackley_hessian(&param);
            let hessian2 = ackley_abc_hessian(&param, 20.0, 0.2, 2.0 * PI);
            for i in 0..3 {
                for j in 0..3 {
                    assert_relative_eq!(hessian1[i][j], hessian2[i][j], epsilon = f64::EPSILON);
                }
            }

            let hessian1 = ackley_hessian_const(&param);
            let hessian2 = ackley_abc_hessian_const(&param, 20.0, 0.2, 2.0 * PI);
            for i in 0..3 {
                for j in 0..3 {
                    assert_relative_eq!(hessian1[i][j], hessian2[i][j], epsilon = f64::EPSILON);
                }
            }
        }
    }

    proptest! {
        #[test]
        fn test_ackley_derivative_finitediff(a in -5.0..5.0,
                                             b in -5.0..5.0,
                                             c in -5.0..5.0,
                                             d in -5.0..5.0,
                                             e in -5.0..5.0,
                                             f in -5.0..5.0,
                                             g in -5.0..5.0,
                                             h in -5.0..5.0) {
            let param = [a, b, c, d, e, f, g, h];
            let derivative = ackley_derivative(&param);
            let derivative_fd = Vec::from(param).central_diff(&|x| ackley(&x));
            // println!("1: {derivative:?} at {a}/{b}");
            // println!("2: {derivative_fd:?} at {a}/{b}");
            for i in 0..derivative.len() {
                assert_relative_eq!(
                    derivative[i],
                    derivative_fd[i],
                    epsilon = 1e-5,
                    max_relative = 1e-2
                );
            }
        }
    }

    proptest! {
        #[test]
        fn test_ackley_derivative_const_finitediff(a in -5.0..5.0,
                                                   b in -5.0..5.0,
                                                   c in -5.0..5.0,
                                                   d in -5.0..5.0,
                                                   e in -5.0..5.0,
                                                   f in -5.0..5.0,
                                                   g in -5.0..5.0,
                                                   h in -5.0..5.0) {
            let param = [a, b, c, d, e, f, g, h];
            let derivative = ackley_derivative_const(&param);
            let derivative_fd = Vec::from(param).central_diff(&|x| ackley(&x));
            // println!("1: {derivative:?} at {a}/{b}");
            // println!("2: {derivative_fd:?} at {a}/{b}");
            for i in 0..derivative.len() {
                assert_relative_eq!(
                    derivative[i],
                    derivative_fd[i],
                    epsilon = 1e-5,
                    max_relative = 1e-2
                );
            }
        }
    }

    proptest! {
        #[test]
        fn test_ackley_hessian_finitediff(a in -5.0..5.0,
                                          b in -5.0..5.0,
                                          c in -5.0..5.0,
                                          d in -5.0..5.0,
                                          e in -5.0..5.0,
                                          f in -5.0..5.0,
                                          g in -5.0..5.0,
                                          h in -5.0..5.0) {
            let param = [a, b, c, d, e, f, g, h];
            let hessian = ackley_hessian(&param);
            let hessian_fd = Vec::from(param).central_hessian(&|x| ackley_derivative(&x));
            // println!("1: {hessian:?} at {a}/{b}/{c}/{d}/{e}/{f}/{g}/{h}");
            // println!("2: {hessian_fd:?} at {a}/{b}/{c}/{d}/{e}/{f}/{g}/{h}");
            for i in 0..hessian.len() {
                for j in 0..hessian.len() {
                    assert_relative_eq!(
                        hessian[i][j],
                        hessian_fd[i][j],
                        epsilon = 1e-5,
                        max_relative = 1e-2
                    );
                }
            }
        }
    }

    proptest! {
        #[test]
        fn test_ackley_hessian_const_finitediff(a in -5.0..5.0,
                                                b in -5.0..5.0,
                                                c in -5.0..5.0,
                                                d in -5.0..5.0,
                                                e in -5.0..5.0,
                                                f in -5.0..5.0,
                                                g in -5.0..5.0,
                                                h in -5.0..5.0) {
            let param = [a, b, c, d, e, f, g, h];
            let hessian = ackley_hessian_const(&param);
            let hessian_fd = Vec::from(param).central_hessian(&|x| ackley_derivative(&x));
            // println!("1: {hessian:?} at {a}/{b}/{c}/{d}/{e}/{f}/{g}/{h}");
            // println!("2: {hessian_fd:?} at {a}/{b}/{c}/{d}/{e}/{f}/{g}/{h}");
            for i in 0..hessian.len() {
                for j in 0..hessian.len() {
                    assert_relative_eq!(
                        hessian[i][j],
                        hessian_fd[i][j],
                        epsilon = 1e-5,
                        max_relative = 1e-2
                    );
                }
            }
        }
    }
}
