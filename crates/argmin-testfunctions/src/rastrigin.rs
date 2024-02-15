// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Rastrigin test function
//!
//! Defined as
//!
//! `f(x_1, x_2, ..., x_n) = a * n + \sum_{i=1}^{n} \left[ x_i^2 - a * cos(2 * pi * x_i) \right]`
//!
//! where `x_i \in [-5.12, 5.12]` and `a = 10`
//!
//! The global minimum is at `f(x_1, x_2, ..., x_n) = f(0, 0, ..., 0) = 0`.

use num::{Float, FromPrimitive};
use std::f64::consts::PI;
use std::iter::Sum;

/// Rastrigin test function
///
/// Defined as
///
/// `f(x_1, x_2, ..., x_n) = a * n + \sum_{i=1}^{n} \left[ x_i^2 - a * cos(2 * pi * x_i) \right]`
///
/// where `x_i \in [-5.12, 5.12]` and `a = 10`
///
/// The global minimum is at `f(x_1, x_2, ..., x_n) = f(0, 0, ..., 0) = 0`.
pub fn rastrigin<T>(param: &[T]) -> T
where
    T: Float + FromPrimitive + Sum,
{
    rastrigin_a(param, T::from_f64(10.0).unwrap())
}

/// Rastrigin test function
///
/// The same as `rastrigin`; however, it allows to set the parameter a.
pub fn rastrigin_a<T>(param: &[T], a: T) -> T
where
    T: Float + FromPrimitive + Sum,
{
    a * T::from_usize(param.len()).unwrap()
        + param
            .iter()
            .map(|&x| x.powi(2) - a * (T::from_f64(2.0 * PI).unwrap() * x).cos())
            .sum()
}

/// Derivative of Rastrigin test function where `a` can be chosen freely
pub fn rastrigin_a_derivative<T>(param: &[T], a: T) -> Vec<T>
where
    T: Float + FromPrimitive + Sum + Into<f64>,
{
    let npi2 = T::from_f64(2.0 * PI).unwrap();
    let n2 = T::from_f64(2.0).unwrap();
    param
        .iter()
        .map(|x| n2 * *x + npi2 * a * T::from_f64(f64::sin((npi2 * *x).into())).unwrap())
        .collect()
}

/// Derivative of Rastrigin test function
pub fn rastrigin_derivative<T>(param: &[T]) -> Vec<T>
where
    T: Float + FromPrimitive + Sum + Into<f64>,
{
    rastrigin_a_derivative(param, T::from_f64(10.0).unwrap())
}

/// Derivative of Rastrigin test function where `a` can be chosen freely
///
/// This is the const generics version, which requires the number of parameters to be known
/// at compile time.
pub fn rastrigin_a_derivative_const<const N: usize, T>(param: &[T; N], a: T) -> [T; N]
where
    T: Float + FromPrimitive + Sum + Into<f64>,
{
    let npi2 = T::from_f64(2.0 * PI).unwrap();
    let n2 = T::from_f64(2.0).unwrap();
    let mut result = [T::from_f64(0.0).unwrap(); N];
    for i in 0..N {
        result[i] =
            n2 * param[i] + npi2 * a * T::from_f64(f64::sin((npi2 * param[i]).into())).unwrap();
    }
    result
}

/// Derivative of Rastrigin test function
///
/// This is the const generics version, which requires the number of parameters to be known
/// at compile time.
pub fn rastrigin_derivative_const<const N: usize, T>(param: &[T; N]) -> [T; N]
where
    T: Float + FromPrimitive + Sum + Into<f64>,
{
    rastrigin_a_derivative_const(param, T::from_f64(10.0).unwrap())
}

/// Hessian of Rastrigin test function where `a` can be chosen freely
pub fn rastrigin_a_hessian<T>(param: &[T], a: T) -> Vec<Vec<T>>
where
    T: Float + FromPrimitive + Sum + Into<f64>,
{
    let npi2 = T::from_f64(2.0 * PI).unwrap();
    let n4pisq = T::from_f64(4.0 * PI.powi(2)).unwrap();
    let n2 = T::from_f64(2.0).unwrap();
    let n0 = T::from_f64(0.0).unwrap();

    let n = param.len();
    let mut hessian = vec![vec![n0; n]; n];

    for i in 0..n {
        hessian[i][i] = n2 + n4pisq * a * T::from_f64(f64::cos((npi2 * param[i]).into())).unwrap();
    }
    hessian
}

/// Hessian of Rastrigin test function
pub fn rastrigin_hessian<T>(param: &[T]) -> Vec<Vec<T>>
where
    T: Float + FromPrimitive + Sum + Into<f64>,
{
    rastrigin_a_hessian(param, T::from_f64(10.0).unwrap())
}

/// Hessian of Rastrigin test function where `a` can be chosen freely
///
/// This is the const generics version, which requires the number of parameters to be known
/// at compile time.
pub fn rastrigin_a_hessian_const<const N: usize, T>(param: &[T], a: T) -> [[T; N]; N]
where
    T: Float + FromPrimitive + Sum + Into<f64>,
{
    let npi2 = T::from_f64(2.0 * PI).unwrap();
    let n4pisq = T::from_f64(4.0 * PI.powi(2)).unwrap();
    let n2 = T::from_f64(2.0).unwrap();
    let n0 = T::from_f64(0.0).unwrap();

    let mut hessian = [[n0; N]; N];

    for i in 0..N {
        hessian[i][i] = n2 + n4pisq * a * T::from_f64(f64::cos((npi2 * param[i]).into())).unwrap();
    }
    hessian
}

/// Hessian of Rastrigin test function
///
/// This is the const generics version, which requires the number of parameters to be known
/// at compile time.
pub fn rastrigin_hessian_const<const N: usize, T>(param: &[T; N]) -> [[T; N]; N]
where
    T: Float + FromPrimitive + Sum + Into<f64>,
{
    rastrigin_a_hessian_const(param, T::from_f64(10.0).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use finitediff::FiniteDiff;
    use proptest::prelude::*;
    use std::{f32, f64};

    #[test]
    fn test_rastrigin_optimum() {
        assert_relative_eq!(rastrigin(&[0.0_f32, 0.0_f32]), 0.0, epsilon = f32::EPSILON);
        assert_relative_eq!(rastrigin(&[0.0_f64, 0.0_f64]), 0.0, epsilon = f64::EPSILON);
    }

    #[test]
    fn test_parameter_a() {
        assert_relative_eq!(
            rastrigin(&[0.0_f32, 0.0_f32]),
            rastrigin_a(&[0.0_f32, 0.0_f32], 10.0),
            epsilon = f32::EPSILON
        );

        assert_relative_eq!(
            rastrigin(&[0.0_f64, 0.0_f64]),
            rastrigin_a(&[0.0_f64, 0.0_f64], 10.0),
            epsilon = f64::EPSILON
        );

        let derivative = rastrigin_derivative(&[1.0_f64, -1.0_f64]);
        let derivative_a = rastrigin_a_derivative(&[1.0_f64, -1.0_f64], 10.0);
        for i in 0..derivative.len() {
            assert_relative_eq!(derivative[i], derivative_a[i], epsilon = f64::EPSILON);
        }

        let derivative = rastrigin_derivative_const(&[1.0_f64, -1.0_f64]);
        let derivative_a = rastrigin_a_derivative_const(&[1.0_f64, -1.0_f64], 10.0);
        for i in 0..derivative.len() {
            assert_relative_eq!(derivative[i], derivative_a[i], epsilon = f64::EPSILON);
        }

        let hessian = rastrigin_hessian(&[1.0_f64, -1.0_f64]);
        let hessian_a = rastrigin_a_hessian(&[1.0_f64, -1.0_f64], 10.0);
        for i in 0..hessian.len() {
            for j in 0..hessian.len() {
                assert_relative_eq!(hessian[i][j], hessian_a[i][j], epsilon = f64::EPSILON);
            }
        }

        let hessian = rastrigin_hessian_const(&[1.0_f64, -1.0_f64]);
        let hessian_a: [[_; 2]; 2] = rastrigin_a_hessian_const(&[1.0_f64, -1.0_f64], 10.0);
        for i in 0..hessian.len() {
            for j in 0..hessian.len() {
                assert_relative_eq!(hessian[i][j], hessian_a[i][j], epsilon = f64::EPSILON);
            }
        }
    }

    #[test]
    fn test_rastrigin_a_derivative_optimum() {
        let derivative = rastrigin_a_derivative(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 10.0);
        for elem in derivative {
            assert_relative_eq!(elem, 0.0, epsilon = std::f64::EPSILON);
        }
    }

    #[test]
    fn test_rastrigin_a_derivative_const_optimum() {
        let derivative =
            rastrigin_a_derivative_const(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 10.0);
        for elem in derivative {
            assert_relative_eq!(elem, 0.0, epsilon = std::f64::EPSILON);
        }
    }

    proptest! {
        #[test]
        fn test_rastrigin_derivative_finitediff(a in -5.12..5.12,
                                                b in -5.12..5.12,
                                                c in -5.12..5.12,
                                                d in -5.12..5.12,
                                                e in -5.12..5.12,
                                                f in -5.12..5.12,
                                                g in -5.12..5.12,
                                                h in -5.12..5.12) {
            let param = [a, b, c, d, e, f, g, h];
            let derivative = rastrigin_derivative(&param);
            let derivative_fd = Vec::from(param).central_diff(&|x| rastrigin(&x));
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
        fn test_rastrigin_derivative_const_finitediff(a in -5.12..5.12,
                                                      b in -5.12..5.12,
                                                      c in -5.12..5.12,
                                                      d in -5.12..5.12,
                                                      e in -5.12..5.12,
                                                      f in -5.12..5.12,
                                                      g in -5.12..5.12,
                                                      h in -5.12..5.12) {
            let param = [a, b, c, d, e, f, g, h];
            let derivative = rastrigin_derivative_const(&param);
            let derivative_fd = Vec::from(param).central_diff(&|x| rastrigin(&x));
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
        fn test_rastrigin_hessian_finitediff(a in -5.12..5.12,
                                             b in -5.12..5.12,
                                             c in -5.12..5.12,
                                             d in -5.12..5.12,
                                             e in -5.12..5.12,
                                             f in -5.12..5.12,
                                             g in -5.12..5.12,
                                             h in -5.12..5.12) {
            let param = [a, b, c, d, e, f, g, h];
            let hessian = rastrigin_hessian(&param);
            let hessian_fd =
                Vec::from(param).forward_hessian(&|x| rastrigin_derivative(&x));
            let n = hessian.len();
            for i in 0..n {
                assert_eq!(hessian[i].len(), n);
                for j in 0..n {
                    assert_relative_eq!(
                        hessian[i][j],
                        hessian_fd[i][j],
                        epsilon = 1e-4,
                        max_relative = 1e-2
                    );
                }
            }
        }
    }

    proptest! {
        #[test]
        fn test_rastrigin_hessian_const_finitediff(a in -5.12..5.12,
                                                   b in -5.12..5.12,
                                                   c in -5.12..5.12,
                                                   d in -5.12..5.12,
                                                   e in -5.12..5.12,
                                                   f in -5.12..5.12,
                                                   g in -5.12..5.12,
                                                   h in -5.12..5.12) {
            let param = [a, b, c, d, e, f, g, h];
            let hessian = rastrigin_hessian_const(&param);
            let hessian_fd =
                Vec::from(param).forward_hessian(&|x| rastrigin_derivative(&x));
            let n = hessian.len();
            for i in 0..n {
                assert_eq!(hessian[i].len(), n);
                for j in 0..n {
                    assert_relative_eq!(
                        hessian[i][j],
                        hessian_fd[i][j],
                        epsilon = 1e-4,
                        max_relative = 1e-2
                    );
                }
            }
        }
    }
}
