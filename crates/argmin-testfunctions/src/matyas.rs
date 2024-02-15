// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Matyas test function
//!
//! Defined as
//!
//! `f(x_1, x_2) = 0.26 * (x_1^2 + x_2^2) - 0.48 * x_1 * x_2`
//!
//! where `x_i \in [-10, 10]`.
//!
//! The global minimum is at `f(x_1, x_2) = f(0, 0) = 0`.

use num::{Float, FromPrimitive};

/// Matyas test function
///
/// Defined as
///
/// `f(x_1, x_2) = 0.26 * (x_1^2 + x_2^2) - 0.48 * x_1 * x_2`
///
/// where `x_i \in [-10, 10]`.
///
/// The global minimum is at `f(x_1, x_2) = f(0, 0) = 0`.
pub fn matyas<T>(param: &[T; 2]) -> T
where
    T: Float + FromPrimitive,
{
    let [x1, x2] = *param;

    let n026 = T::from_f64(0.26).unwrap();
    let n048 = T::from_f64(0.48).unwrap();

    n026 * (x1.powi(2) + x2.powi(2)) - n048 * x1 * x2
}

/// Derivative of Matyas test function
pub fn matyas_derivative<T>(param: &[T; 2]) -> [T; 2]
where
    T: Float + FromPrimitive,
{
    let [x1, x2] = *param;

    let n0_52 = T::from_f64(0.52).unwrap();
    let n0_48 = T::from_f64(0.48).unwrap();

    [n0_52 * x1 - n0_48 * x2, n0_52 * x2 - n0_48 * x1]
}

/// Hessian of Matyas test function
///
/// Returns [[0.52, -0.48], [-0.48, 0.52]] for any input.
pub fn matyas_hessian<T>(_param: &[T; 2]) -> [[T; 2]; 2]
where
    T: Float + FromPrimitive,
{
    let n0_52 = T::from_f64(0.52).unwrap();
    let n0_48 = T::from_f64(0.48).unwrap();

    [[n0_52, -n0_48], [-n0_48, n0_52]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use finitediff::FiniteDiff;
    use proptest::prelude::*;
    use std::{f32, f64};

    #[test]
    fn test_matyas_optimum() {
        assert_relative_eq!(matyas(&[0_f32, 0_f32]), 0.0, epsilon = f32::EPSILON);
        assert_relative_eq!(matyas(&[0_f64, 0_f64]), 0.0, epsilon = f64::EPSILON);

        let deriv = matyas_derivative(&[0.0, 0.0]);
        for i in 0..2 {
            assert_relative_eq!(deriv[i], 0.0, epsilon = f64::EPSILON);
        }
    }

    proptest! {
        #[test]
        fn test_matyas_derivative(a in -10.0..10.0, b in -10.0..10.0) {
            let param = [a, b];
            let derivative = matyas_derivative(&param);
            let derivative_fd = Vec::from(param).central_diff(&|x| matyas(&[x[0], x[1]]));
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
        fn test_matyas_hessian(a in -10.0..10.0, b in -10.0..10.0) {
            let param = [a, b];
            let hessian = matyas_hessian(&param);
            let hessian_fd = [[0.52, -0.48], [-0.48, 0.52]];
            let n = hessian.len();
            for i in 0..n {
                assert_eq!(hessian[i].len(), n);
                for j in 0..n {
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
