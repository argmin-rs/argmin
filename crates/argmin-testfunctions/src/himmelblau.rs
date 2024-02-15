// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Himmelblau test function
//!
//! Defined as
//!
//! `f(x_1, x_2) = (x_1^2 + x_2 - 11)^2 + (x_1 + x_2^2 - 7)^2`
//!
//! where `x_i \in [-5, 5]`.
//!
//! The global minima are at
//!  * `f(x_1, x_2) = f(3, 2) = 0`.
//!  * `f(x_1, x_2) = f(-2.805118, 3.131312) = 0`.
//!  * `f(x_1, x_2) = f(-3.779310, -3.283186) = 0`.
//!  * `f(x_1, x_2) = f(3.584428, -1.848126) = 0`.

use num::{Float, FromPrimitive};

/// Himmelblau test function
///
/// Defined as
///
/// `f(x_1, x_2) = (x_1^2 + x_2 - 11)^2 + (x_1 + x_2^2 - 7)^2`
///
/// where `x_i \in [-5, 5]`.
///
/// The global minima are at
///  * `f(x_1, x_2) = f(3, 2) = 0`.
///  * `f(x_1, x_2) = f(-2.805118, 3.131312) = 0`.
///  * `f(x_1, x_2) = f(-3.779310, -3.283186) = 0`.
///  * `f(x_1, x_2) = f(3.584428, -1.848126) = 0`.
pub fn himmelblau<T>(param: &[T; 2]) -> T
where
    T: Float + FromPrimitive,
{
    let [x1, x2] = *param;
    let n7 = T::from_f64(7.0).unwrap();
    let n11 = T::from_f64(11.0).unwrap();
    (x1.powi(2) + x2 - n11).powi(2) + (x1 + x2.powi(2) - n7).powi(2)
}

/// Derivative of Himmelblau test function
pub fn himmelblau_derivative<T>(param: &[T; 2]) -> [T; 2]
where
    T: Float + FromPrimitive,
{
    let [x1, x2] = *param;

    let n2 = T::from_f64(2.0).unwrap();
    let n4 = T::from_f64(4.0).unwrap();
    let n7 = T::from_f64(7.0).unwrap();
    let n11 = T::from_f64(11.0).unwrap();

    [
        n4 * x1 * (x1.powi(2) + x2 - n11) + n2 * (x1 + x2.powi(2) - n7),
        n4 * x2 * (x2.powi(2) + x1 - n7) + n2 * (x2 + x1.powi(2) - n11),
    ]
}

/// Hessian of Himmelblau test function
pub fn himmelblau_hessian<T>(param: &[T; 2]) -> [[T; 2]; 2]
where
    T: Float + FromPrimitive,
{
    let [x1, x2] = *param;

    let n2 = T::from_f64(2.0).unwrap();
    let n4 = T::from_f64(4.0).unwrap();
    let n7 = T::from_f64(7.0).unwrap();
    let n8 = T::from_f64(8.0).unwrap();
    let n11 = T::from_f64(11.0).unwrap();

    let offdiag = n4 * (x1 + x2);

    [
        [n4 * (x1.powi(2) + x2 - n11) + n8 * x1.powi(2) + n2, offdiag],
        [offdiag, n4 * (x2.powi(2) + x1 - n7) + n8 * x2.powi(2) + n2],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use finitediff::FiniteDiff;
    use proptest::prelude::*;
    use std::{f32, f64};

    #[test]
    fn test_himmelblau_optimum() {
        assert_relative_eq!(himmelblau(&[3.0_f32, 2.0_f32]), 0.0, epsilon = f32::EPSILON);
        assert_relative_eq!(
            himmelblau(&[-2.805118_f32, 3.131312_f32]),
            0.0,
            epsilon = f32::EPSILON
        );
        assert_relative_eq!(
            himmelblau(&[-3.779310_f32, -3.283186_f32]),
            0.0,
            epsilon = f32::EPSILON
        );
        assert_relative_eq!(
            himmelblau(&[3.584428_f32, -1.848126_f32]),
            0.0,
            epsilon = f32::EPSILON
        );

        // Since we don't know the 64bit location of the minima,the f64 version cannot be reliably
        // tested without allowing an error larger than f64::EPSILON.
        assert_relative_eq!(himmelblau(&[3.0_f64, 2.0_f64]), 0.0, epsilon = f64::EPSILON);
        assert_relative_eq!(
            himmelblau(&[-2.805118_f64, 3.131312_f64]),
            0.0,
            epsilon = f32::EPSILON.into()
        );
        assert_relative_eq!(
            himmelblau(&[-3.779310_f64, -3.283186_f64]),
            0.0,
            epsilon = f32::EPSILON.into()
        );
        assert_relative_eq!(
            himmelblau(&[3.584428_f64, -1.848126_f64]),
            0.0,
            epsilon = f32::EPSILON.into()
        );

        let deriv = himmelblau_derivative(&[3.0_f32, 2.0_f32]);
        for i in 0..2 {
            assert_relative_eq!(deriv[i], 0.0, epsilon = f32::EPSILON);
        }

        let deriv = himmelblau_derivative(&[-2.805118_f32, 3.131312_f32]);
        for i in 0..2 {
            assert_relative_eq!(deriv[i], 0.0, epsilon = 1e-4);
        }

        let deriv = himmelblau_derivative(&[-3.779310_f64, -3.283186_f64]);
        for i in 0..2 {
            assert_relative_eq!(deriv[i], 0.0, epsilon = 1e-4);
        }

        let deriv = himmelblau_derivative(&[3.584428_f64, -1.848126_f64]);
        for i in 0..2 {
            assert_relative_eq!(deriv[i], 0.0, epsilon = 1e-4);
        }
    }

    proptest! {
        #[test]
        fn test_himmelblau_derivative_finitediff(a in -5.0..5.0, b in -5.0..5.0) {
            let param = [a, b];
            let derivative = himmelblau_derivative(&param);
            let derivative_fd = Vec::from(param).central_diff(&|x| himmelblau(&[x[0], x[1]]));
            for i in 0..derivative.len() {
                assert_relative_eq!(
                    derivative[i],
                    derivative_fd[i],
                    epsilon = 1e-4,
                    max_relative = 1e-2
                );
            }
        }
    }

    proptest! {
        #[test]
        fn test_himmelblau_hessian_finitediff(a in -5.0..5.0, b in -5.0..5.0) {
            let param = [a, b];
            let hessian = himmelblau_hessian(&param);
            let hessian_fd =
                Vec::from(param).central_hessian(&|x| himmelblau_derivative(&[x[0], x[1]]).to_vec());
            let n = hessian.len();
            // println!("1: {hessian:?} at {a}/{b}");
            // println!("2: {hessian_fd:?} at {a}/{b}");
            for i in 0..n {
                assert_eq!(hessian[i].len(), n);
                for j in 0..n {
                    if hessian_fd[i][j].is_finite() {
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
}
