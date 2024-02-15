// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Three-hump camel test function
//!
//! Defined as
//!
//! `f(x_1, x_2) = 2*x_1^2 - 1.05*x_1^4 + x_1^6/6 + x_1*x_2 + x_2^2`
//!
//! where `x_i \in [-5, 5]`.
//!
//! The global minimum is at `f(x_1, x_2) = f(0, 0) = 0`.

use num::{Float, FromPrimitive};

/// Three-hump camel test function
///
/// Defined as
///
/// `f(x_1, x_2) = 2*x_1^2 - 1.05*x_1^4 + x_1^6/6 + x_1*x_2 + x_2^2`
///
/// where `x_i \in [-5, 5]`.
///
/// The global minimum is at `f(x_1, x_2) = f(0, 0) = 0`.
pub fn threehumpcamel<T>(param: &[T; 2]) -> T
where
    T: Float + FromPrimitive,
{
    let [x1, x2] = *param;

    T::from_f64(2.0).unwrap() * x1.powi(2) - T::from_f64(1.05).unwrap() * x1.powi(4)
        + x1.powi(6) / T::from_f64(6.0).unwrap()
        + x1 * x2
        + x2.powi(2)
}

/// Derivative of Three-hump camel test function
pub fn threehumpcamel_derivative<T>(param: &[T; 2]) -> [T; 2]
where
    T: Float + FromPrimitive,
{
    let [x1, x2] = *param;

    let n2 = T::from_f64(2.0).unwrap();
    let n4 = T::from_f64(4.0).unwrap();
    let n4_2 = T::from_f64(4.2).unwrap();

    [x1.powi(5) - n4_2 * x1.powi(3) + n4 * x1 + x2, n2 * x2 + x1]
}

/// Hessian of Three-hump camel test function
pub fn threehumpcamel_hessian<T>(param: &[T; 2]) -> [[T; 2]; 2]
where
    T: Float + FromPrimitive,
{
    let [x1, _] = *param;

    let n1 = T::from_f64(1.0).unwrap();
    let n2 = T::from_f64(2.0).unwrap();
    let n4 = T::from_f64(4.0).unwrap();
    let n5 = T::from_f64(5.0).unwrap();
    let n12_6 = T::from_f64(12.6).unwrap();

    let a = n5 * x1.powi(4) - n12_6 * x1.powi(2) + n4;
    let b = n2;
    let offdiag = n1;

    [[a, offdiag], [offdiag, b]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use finitediff::FiniteDiff;
    use proptest::prelude::*;
    use std::{f32, f64};

    #[test]
    fn test_threehumpcamel_optimum() {
        assert_relative_eq!(
            threehumpcamel(&[0.0_f32, 0.0_f32]),
            0.0,
            epsilon = f32::EPSILON
        );
        assert_relative_eq!(
            threehumpcamel(&[0.0_f64, 0.0_f64]),
            0.0,
            epsilon = f64::EPSILON
        );

        let deriv = threehumpcamel_derivative(&[0.0, 0.0]);
        for i in 0..2 {
            assert_relative_eq!(deriv[i], 0.0, epsilon = f64::EPSILON);
        }
    }

    proptest! {
        #[test]
        fn test_threehumpcamel_derivative(a in -5.0..5.0, b in -5.0..5.0) {
            let param = [a, b];
            let derivative = threehumpcamel_derivative(&param);
            let derivative_fd = Vec::from(param).central_diff(&|x| threehumpcamel(&[x[0], x[1]]));
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
        fn test_threehumpcamel_hessian_finitediff(a in -5.0..5.0, b in -5.0..5.0) {
            let param = [a, b];
            let hessian = threehumpcamel_hessian(&param);
            let hessian_fd =
                Vec::from(param).central_hessian(&|x| threehumpcamel_derivative(&[x[0], x[1]]).to_vec());
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
