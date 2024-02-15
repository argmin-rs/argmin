// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Sphere function
//!
//! Defined as
//!
//! `f(x) = \sum_{i=1}^n x_i^2`
//!
//! where `x_i \in (-\infty, \infty)`
//!
//! The minimum is at `f(x_1, x_2, ..., x_n) = f(0, 0, ..., 0) = 0`.

use num::{Float, FromPrimitive};
use std::iter::Sum;

/// Sphere test function
///
/// Defined as
///
/// `f(x_1, x_2, ..., x_n) = \sum_{i=1}^n x_i^2
///
/// where `x_i \in (-\infty, \infty)` and `n > 0`.
///
/// The global minimum is at `f(x_1, x_2, ..., x_n) = f(0, 0, ..., 0) = 0`.
pub fn sphere<T>(param: &[T]) -> T
where
    T: Float + FromPrimitive + Sum,
{
    param.iter().map(|x| x.powi(2)).sum()
}

/// Derivative of sphere test function
///
/// Defined as
///
/// `f(x_1, x_2, ..., x_n) = (2 * x_1, 2 * x_2, ... 2 * x_n)`
///
/// where `x_i \in (-\infty, \infty)` and `n > 0`.
pub fn sphere_derivative<T>(param: &[T]) -> Vec<T>
where
    T: Float + FromPrimitive,
{
    let num2 = T::from_f64(2.0).unwrap();
    param.iter().map(|x| num2 * *x).collect()
}

/// Derivative of sphere test function
///
/// Defined as
///
/// `f(x_1, x_2, ..., x_n) = (2 * x_1, 2 * x_2, ... 2 * x_n)`
///
/// where `x_i \in (-\infty, \infty)` and `n > 0`.
///
/// This is the const generics version, which requires the number of parameters to be known
/// at compile time.
pub fn sphere_derivative_const<const N: usize, T>(param: &[T; N]) -> [T; N]
where
    T: Float + FromPrimitive,
{
    let num2 = T::from_f64(2.0).unwrap();
    let mut deriv = [T::from_f64(0.0).unwrap(); N];
    for i in 0..N {
        deriv[i] = num2 * param[i];
    }
    deriv
}

/// Hessian of sphere test function
pub fn sphere_hessian<T>(param: &[T]) -> Vec<Vec<T>>
where
    T: Float + FromPrimitive,
{
    let n = param.len();
    let mut hessian = vec![vec![T::from_f64(0.0).unwrap(); n]; n];
    for (i, row) in hessian.iter_mut().enumerate().take(n) {
        row[i] = T::from_f64(2.0).unwrap();
    }
    hessian
}

/// Hessian of sphere test function
///
/// This is the const generics version, which requires the number of parameters to be known
/// at compile time.
pub fn sphere_hessian_const<const N: usize, T>(_param: &[T; N]) -> [[T; N]; N]
where
    T: Float + FromPrimitive,
{
    let mut hessian = [[T::from_f64(0.0).unwrap(); N]; N];
    for (i, row) in hessian.iter_mut().enumerate().take(N) {
        row[i] = T::from_f64(2.0).unwrap();
    }
    hessian
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use finitediff::FiniteDiff;
    use proptest::prelude::*;
    use std;

    #[test]
    fn test_sphere_optimum() {
        assert_relative_eq!(
            sphere(&[0.0_f32, 0.0_f32]),
            0.0,
            epsilon = std::f32::EPSILON
        );
        assert_relative_eq!(
            sphere(&[0.0_f64, 0.0_f64]),
            0.0,
            epsilon = std::f64::EPSILON
        );
    }

    proptest! {
        #[test]
        fn test_sphere(a in -10.0..10.0,
                       b in -10.0..10.0,
                       c in -10.0..10.0,
                       d in -10.0..10.0,
                       e in -10.0..10.0,
                       f in -10.0..10.0,
                       g in -10.0..10.0,
                       h in -10.0..10.0) {
            let param: [f64; 8] = [a, b, c, d, e, f, g, h];
            let v1 = sphere(&param);
            let v2 = a.powi(2) + b.powi(2) + c.powi(2) + d.powi(2) + e.powi(2) + f.powi(2) + g.powi(2) + h.powi(2);
            assert_relative_eq!(v1, v2, epsilon = std::f64::EPSILON);
        }
    }

    proptest! {
        #[test]
        fn test_sphere_derivative(a in -10.0..10.0,
                                  b in -10.0..10.0,
                                  c in -10.0..10.0,
                                  d in -10.0..10.0,
                                  e in -10.0..10.0,
                                  f in -10.0..10.0,
                                  g in -10.0..10.0,
                                  h in -10.0..10.0) {
            let param = [a, b, c, d, e, f, g, h];
            let derivative = sphere_derivative(&param);
            let derivative_fd =
                [2.0 * a, 2.0 * b, 2.0 * c, 2.0 * d, 2.0 * e, 2.0 * f, 2.0 * g, 2.0 * h];
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
        fn test_sphere_derivative_const(a in -10.0..10.0,
                                        b in -10.0..10.0,
                                        c in -10.0..10.0,
                                        d in -10.0..10.0,
                                        e in -10.0..10.0,
                                        f in -10.0..10.0,
                                        g in -10.0..10.0,
                                        h in -10.0..10.0) {
            let param = [a, b, c, d, e, f, g, h];
            let derivative = sphere_derivative_const(&param);
            let derivative_fd =
                [2.0 * a, 2.0 * b, 2.0 * c, 2.0 * d, 2.0 * e, 2.0 * f, 2.0 * g, 2.0 * h];
            for i in 0..derivative.len() {
                assert_relative_eq!(
                    derivative[i],
                    derivative_fd[i],
                    epsilon = 1e-5,
                    max_relative = 1e-2,
                );
            }
        }
    }

    proptest! {
        #[test]
        fn test_sphere_derivative_finitediff(a in -10.0..10.0,
                                             b in -10.0..10.0,
                                             c in -10.0..10.0,
                                             d in -10.0..10.0,
                                             e in -10.0..10.0,
                                             f in -10.0..10.0,
                                             g in -10.0..10.0,
                                             h in -10.0..10.0) {
            let param = [a, b, c, d, e, f, g, h];
            let derivative = sphere_derivative(&param);
            let derivative_fd = Vec::from(param).central_diff(&|x| sphere(&x));
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
        fn test_sphere_derivative_const_finitediff(a in -10.0..10.0,
                                                   b in -10.0..10.0,
                                                   c in -10.0..10.0,
                                                   d in -10.0..10.0,
                                                   e in -10.0..10.0,
                                                   f in -10.0..10.0,
                                                   g in -10.0..10.0,
                                                   h in -10.0..10.0) {
            let param = [a, b, c, d, e, f, g, h];
            let derivative = sphere_derivative_const(&param);
            let derivative_fd = Vec::from(param).central_diff(&|x| sphere(&x));
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
        fn test_sphere_hessian(a in -10.0..10.0,
                                  b in -10.0..10.0,
                                  c in -10.0..10.0,
                                  d in -10.0..10.0,
                                  e in -10.0..10.0,
                                  f in -10.0..10.0,
                                  g in -10.0..10.0,
                                  h in -10.0..10.0) {
            let param = [a, b, c, d, e, f, g, h];
            let hessian = sphere_hessian(&param);
            for i in 0..hessian.len() {
                for j in 0..hessian.len() {
                    if i == j {
                        assert_relative_eq!(hessian[i][j], 2.0, epsilon = std::f64::EPSILON);
                    } else {
                        assert_relative_eq!(hessian[i][j], 0.0, epsilon = std::f64::EPSILON);
                    }
                }
            }
        }
    }

    proptest! {
        #[test]
        fn test_sphere_hessian_const(a in -10.0..10.0,
                                     b in -10.0..10.0,
                                     c in -10.0..10.0,
                                     d in -10.0..10.0,
                                     e in -10.0..10.0,
                                     f in -10.0..10.0,
                                     g in -10.0..10.0,
                                     h in -10.0..10.0) {
            let param = [a, b, c, d, e, f, g, h];
            let hessian = sphere_hessian_const(&param);
            for i in 0..hessian.len() {
                for j in 0..hessian.len() {
                    if i == j {
                        assert_relative_eq!(hessian[i][j], 2.0, epsilon = std::f64::EPSILON);
                    } else {
                        assert_relative_eq!(hessian[i][j], 0.0, epsilon = std::f64::EPSILON);
                    }
                }
            }
        }
    }

    proptest! {
        #[test]
        fn test_sphere_hessian_finitediff(a in -10.0..10.0,
                                          b in -10.0..10.0,
                                          c in -10.0..10.0,
                                          d in -10.0..10.0,
                                          e in -10.0..10.0,
                                          f in -10.0..10.0,
                                          g in -10.0..10.0,
                                          h in -10.0..10.0) {
            let param = [a, b, c, d, e, f, g, h];
            let hessian = sphere_hessian(&param);
            let hessian_fd = Vec::from(param).central_hessian(&|x| sphere_derivative(&x));
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
        fn test_sphere_hessian_const_finitediff(a in -10.0..10.0,
                                                b in -10.0..10.0,
                                                c in -10.0..10.0,
                                                d in -10.0..10.0,
                                                e in -10.0..10.0,
                                                f in -10.0..10.0,
                                                g in -10.0..10.0,
                                                h in -10.0..10.0) {
            let param = [a, b, c, d, e, f, g, h];
            let hessian = sphere_hessian_const(&param);
            let hessian_fd = Vec::from(param).central_hessian(&|x| sphere_derivative(&x));
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
