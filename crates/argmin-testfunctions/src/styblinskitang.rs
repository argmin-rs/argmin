// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Styblinski-Tang test function
//!
//! Defined as
//!
//! `f(x_1, x_2, ..., x_n) = 1/2 * \sum_{i=1}^{n} \left[ x_i^4 - 16 * x_i^2 + 5 * x_i \right]`
//!
//! where `x_i \in [-5, 5]`.
//!
//! The global minimum is at `f(x_1, x_2, ..., x_n) = f(-2.903534, -2.903534, ..., -2.903534) =
//! -39.16616*n`.

use num::{Float, FromPrimitive};
use std::iter::Sum;

/// Styblinski-Tang test function
///
/// Defined as
///
/// `f(x_1, x_2, ..., x_n) = 1/2 * \sum_{i=1}^{n} \left[ x_i^4 - 16 * x_i^2 + 5 * x_i \right]`
///
/// where `x_i \in [-5, 5]`.
///
/// The global minimum is at `f(x_1, x_2, ..., x_n) = f(-2.903534, -2.903534, ..., -2.903534) =
/// -39.16616*n`.
pub fn styblinski_tang<T>(param: &[T]) -> T
where
    T: Float + FromPrimitive + Sum,
{
    T::from_f64(0.5).unwrap()
        * param
            .iter()
            .map(|x| {
                x.powi(4) - T::from_f64(16.0).unwrap() * x.powi(2) + T::from_f64(5.0).unwrap() * *x
            })
            .sum()
}

/// Derivative of Styblinski-Tang test function
pub fn styblinski_tang_derivative<T>(param: &[T]) -> Vec<T>
where
    T: Float + FromPrimitive + Sum,
{
    let n2 = T::from_f64(2.0).unwrap();
    let n2_5 = T::from_f64(2.5).unwrap();
    let n16 = T::from_f64(16.0).unwrap();

    param
        .iter()
        .map(|x| n2 * x.powi(3) - n16 * *x + n2_5)
        .collect()
}

/// Derivative of Styblinski-Tang test function
///
/// This is the const generics version, which requires the number of parameters to be known
/// at compile time.
pub fn styblinski_tang_derivative_const<const N: usize, T>(param: &[T; N]) -> [T; N]
where
    T: Float + FromPrimitive + Sum,
{
    let n0 = T::from_f64(0.0).unwrap();
    let n2 = T::from_f64(2.0).unwrap();
    let n2_5 = T::from_f64(2.5).unwrap();
    let n16 = T::from_f64(16.0).unwrap();

    let mut out = [n0; N];

    param
        .iter()
        .zip(out.iter_mut())
        .map(|(x, o)| *o = n2 * x.powi(3) - n16 * *x + n2_5)
        .count();

    out
}

/// Hessian of Styblinski-Tang test function
pub fn styblinski_tang_hessian<T>(param: &[T]) -> Vec<Vec<T>>
where
    T: Float + FromPrimitive + Sum,
{
    let n0 = T::from_f64(0.0).unwrap();
    let n6 = T::from_f64(6.0).unwrap();
    let n16 = T::from_f64(16.0).unwrap();

    let n = param.len();
    let mut out = vec![vec![n0; n]; n];

    param
        .iter()
        .enumerate()
        .map(|(i, x)| out[i][i] = n6 * x.powi(2) - n16)
        .count();

    out
}

/// Hessian of Styblinski-Tang test function
///
/// This is the const generics version, which requires the number of parameters to be known
/// at compile time.
pub fn styblinski_tang_hessian_const<const N: usize, T>(param: &[T; N]) -> [[T; N]; N]
where
    T: Float + FromPrimitive + Sum,
{
    let n0 = T::from_f64(0.0).unwrap();
    let n6 = T::from_f64(6.0).unwrap();
    let n16 = T::from_f64(16.0).unwrap();

    let mut out = [[n0; N]; N];

    param
        .iter()
        .enumerate()
        .map(|(i, x)| out[i][i] = n6 * x.powi(2) - n16)
        .count();

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use finitediff::FiniteDiff;
    use proptest::prelude::*;
    use std::f32;

    #[test]
    fn test_styblinski_tang_optimum() {
        assert_relative_eq!(
            styblinski_tang(&[-2.903534_f32, -2.903534_f32, -2.903534_f32]),
            -117.49849,
            epsilon = f32::EPSILON
        );
        assert_relative_eq!(
            styblinski_tang(&[-2.903534_f64, -2.903534_f64, -2.903534_f64]),
            -117.4984971113142,
            epsilon = f64::EPSILON
        );

        let deriv = styblinski_tang_derivative(&[-2.903534_f64, -2.903534_f64, -2.903534_f64]);
        for i in 0..3 {
            assert_relative_eq!(deriv[i], 0.0, epsilon = 1e-5, max_relative = 1e-2);
        }
    }

    proptest! {
        #[test]
        fn test_styblinski_tang_derivative_finitediff(a in -5.0..5.0,
                                                      b in -5.0..5.0,
                                                      c in -5.0..5.0,
                                                      d in -5.0..5.0,
                                                      e in -5.0..5.0,
                                                      f in -5.0..5.0,
                                                      g in -5.0..5.0,
                                                      h in -5.0..5.0) {
            let param = [a, b, c, d, e, f, g, h];
            let derivative = styblinski_tang_derivative(&param);
            let derivative_fd = Vec::from(param).central_diff(&|x| styblinski_tang(&x));
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
        fn test_styblinski_tang_derivative_const_finitediff(a in -5.0..5.0,
                                                            b in -5.0..5.0,
                                                            c in -5.0..5.0,
                                                            d in -5.0..5.0,
                                                            e in -5.0..5.0,
                                                            f in -5.0..5.0,
                                                            g in -5.0..5.0,
                                                            h in -5.0..5.0) {
            let param = [a, b, c, d, e, f, g, h];
            let derivative = styblinski_tang_derivative_const(&param);
            let derivative_fd = Vec::from(param).central_diff(&|x| styblinski_tang(&x));
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
        fn test_styblinski_tang_hessian_finitediff(a in -5.0..5.0,
                                                   b in -5.0..5.0,
                                                   c in -5.0..5.0,
                                                   d in -5.0..5.0,
                                                   e in -5.0..5.0,
                                                   f in -5.0..5.0,
                                                   g in -5.0..5.0,
                                                   h in -5.0..5.0) {
            let param = [a, b, c, d, e, f, g, h];
            let derivative = styblinski_tang_hessian(&param);
            let derivative_fd = Vec::from(param).central_hessian(&|x| styblinski_tang_derivative(&x));
            for i in 0..derivative.len() {
                for j in 0..derivative[i].len() {
                    assert_relative_eq!(
                        derivative[i][j],
                        derivative_fd[i][j],
                        epsilon = 1e-5,
                        max_relative = 1e-2
                    );
                }
            }
        }
    }

    proptest! {
        #[test]
        fn test_styblinski_tang_hessian_const_finitediff(a in -5.0..5.0,
                                                         b in -5.0..5.0,
                                                         c in -5.0..5.0,
                                                         d in -5.0..5.0,
                                                         e in -5.0..5.0,
                                                         f in -5.0..5.0,
                                                         g in -5.0..5.0,
                                                         h in -5.0..5.0) {
            let param = [a, b, c, d, e, f, g, h];
            let derivative = styblinski_tang_hessian_const(&param);
            let derivative_fd = Vec::from(param).central_hessian(&|x| styblinski_tang_derivative(&x));
            for i in 0..derivative.len() {
                for j in 0..derivative[i].len() {
                    assert_relative_eq!(
                        derivative[i][j],
                        derivative_fd[i][j],
                        epsilon = 1e-5,
                        max_relative = 1e-2
                    );
                }
            }
        }
    }
}
