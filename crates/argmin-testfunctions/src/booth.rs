// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Booth test function
//!
//! Defined as
//!
//! `f(x_1, x_2) = (x_1 + 2*x_2 - 7)^2 + (2*x_1 + x_2 - 5)^2`
//!
//! where `x_i \in [-10, 10]`.
//!
//! The global minimum is at `f(x_1, x_2) = f(1, 3) = 0`.

use num::{Float, FromPrimitive};

/// Booth test function
///
/// Defined as
///
/// `f(x_1, x_2) = (x_1 + 2*x_2 - 7)^2 + (2*x_1 + x_2 - 5)^2
///
/// where `x_i \in [-10, 10]`.
///
/// The global minimum is at `f(x_1, x_2) = f(1, 3) = 0`.
pub fn booth<T>(param: &[T; 2]) -> T
where
    T: Float + FromPrimitive,
{
    let n2 = T::from_f64(2.0).unwrap();
    let n5 = T::from_f64(5.0).unwrap();
    let n7 = T::from_f64(7.0).unwrap();

    let [x1, x2] = *param;
    (x1 + n2 * x2 - n7).powi(2) + (n2 * x1 + x2 - n5).powi(2)
}

/// Derivative of Booth test function
pub fn booth_derivative<T>(param: &[T; 2]) -> [T; 2]
where
    T: Float + FromPrimitive,
{
    let n8 = T::from_f64(8.0).unwrap();
    let n10 = T::from_f64(10.0).unwrap();
    let n34 = T::from_f64(34.0).unwrap();
    let n38 = T::from_f64(38.0).unwrap();

    let [x1, x2] = *param;

    [n10 * x1 + n8 * x2 - n34, n8 * x1 + n10 * x2 - n38]
}

/// Hessian of Booth test function
///
/// Returns [[10, 8], [8, 10]] for every input.
pub fn booth_hessian<T>(_param: &[T; 2]) -> [[T; 2]; 2]
where
    T: Float + FromPrimitive,
{
    let n8 = T::from_f64(8.0).unwrap();
    let n10 = T::from_f64(10.0).unwrap();

    [[n10, n8], [n8, n10]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use finitediff::FiniteDiff;
    use proptest::prelude::*;
    use std::{f32, f64};

    #[test]
    fn test_booth_optimum() {
        assert_relative_eq!(booth(&[1_f32, 3_f32]), 0.0, epsilon = f32::EPSILON);
        assert_relative_eq!(booth(&[1_f64, 3_f64]), 0.0, epsilon = f64::EPSILON);

        let deriv = booth_derivative(&[1.0, 3.0]);
        for i in 0..2 {
            assert_relative_eq!(deriv[i], 0.0, epsilon = f64::EPSILON);
        }
    }

    proptest! {
        #[test]
        fn test_booth_derivative_finitediff(a in -10.0..10.0, b in -10.0..10.0) {
            let param = [a, b];
            let derivative = booth_derivative(&param);
            let derivative_fd = Vec::from(param).central_diff(&|x| booth(&[x[0], x[1]]));
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
        fn test_booth_hessian(a in -10.0..10.0, b in -10.0..10.0) {
            let param = [a, b];
            let hessian = booth_hessian(&param);
            let hessian_fd = [[10.0, 8.0], [8.0, 10.0]];
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
