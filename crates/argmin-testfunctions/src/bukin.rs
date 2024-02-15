// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Bukin test function No. 6
//!
//! Defined as
//!
//! `f(x_1, x_2) = 100*\sqrt{|x_2 - 0.01*x_1^2|} + 0.01 * |x_1 + 10|`
//!
//! where `x_1 \in [-15, -5]` and `x_2 \in [-3, 3]`.
//!
//! The global minimum is at `f(x_1, x_2) = f(-10, 1) = 0`.

use num::{Float, FromPrimitive};

/// Bukin test function No. 6
///
/// Defined as
///
/// `f(x_1, x_2) = 100*\sqrt{|x_2 - 0.01*x_1^2|} + 0.01 * |x_1 + 10|`
///
/// where `x_1 \in [-15, -5]` and `x_2 \in [-3, 3]`.
///
/// The global minimum is at `f(x_1, x_2) = f(-10, 1) = 0`.
pub fn bukin_n6<T>(param: &[T; 2]) -> T
where
    T: Float + FromPrimitive,
{
    let [x1, x2] = *param;
    let n001 = T::from_f64(0.01).unwrap();
    let n10 = T::from_f64(10.0).unwrap();
    let n100 = T::from_f64(100.0).unwrap();
    n100 * (x2 - n001 * x1.powi(2)).abs().sqrt() + n001 * (x1 + n10).abs()
}

/// Derivative of Bukin test function No. 6
pub fn bukin_n6_derivative<T>(param: &[T; 2]) -> [T; 2]
where
    T: Float + FromPrimitive,
{
    let [x1, x2] = *param;

    let n0 = T::from_f64(0.0).unwrap();
    let n0_01 = T::from_f64(0.01).unwrap();
    let n10 = T::from_f64(10.0).unwrap();
    let n50 = T::from_f64(50.0).unwrap();
    let eps = T::epsilon();

    let denom = (x2 - n0_01 * x1.powi(2)).abs().powi(3).sqrt();
    let tmp = x2 - n0_01 * x1.powi(2);

    if denom.abs() <= eps {
        // Deriviative is actually not defined at optimum. Therefore, as soon as we get close,
        // we'll set the derivative to 0
        [n0, n0]
    } else {
        [
            n0_01 * (x1 + n10).signum() - x1 * tmp / denom,
            n50 * tmp / denom,
        ]
    }
}

/// Hessian of Bukin test function No. 6
pub fn bukin_n6_hessian<T>(param: &[T; 2]) -> [[T; 2]; 2]
where
    T: Float + FromPrimitive,
{
    let [x1, x2] = *param;

    let n0 = T::from_f64(0.0).unwrap();
    let n0_01 = T::from_f64(0.01).unwrap();
    let n0_02 = T::from_f64(0.02).unwrap();
    let n0_0001 = T::from_f64(0.0001).unwrap();
    let n0_5 = T::from_f64(0.5).unwrap();
    let n25 = T::from_f64(25.0).unwrap();
    let eps = T::epsilon() * T::from_f64(1e-4).unwrap();

    let tmp = x2 - n0_01 * x1.powi(2);
    let denom = tmp.abs().powi(7).sqrt();

    if denom.abs() <= eps {
        // Hessian is actually not defined at optimum. Therefore, as soon as we get close,
        // we'll set the Hessian to 0
        [[n0, n0], [n0, n0]]
    } else {
        let offdiag = n0_5 * x1 * tmp.powi(2) / denom;
        [
            [
                x2 * (-n0_0001 * x1.powi(4) + n0_02 * x2 * x1.powi(2) - x2.powi(2)) / denom,
                offdiag,
            ],
            [offdiag, -n25 * tmp.powi(2) / denom],
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use finitediff::FiniteDiff;
    use proptest::prelude::*;
    use std::{f32, f64};

    #[test]
    fn test_bukin_n6_optimum() {
        assert_relative_eq!(bukin_n6(&[-10_f32, 1_f32]), 0.0, epsilon = f32::EPSILON);
        assert_relative_eq!(bukin_n6(&[-10_f64, 1_f64]), 0.0, epsilon = f64::EPSILON);

        let deriv = bukin_n6_derivative(&[-10_f64, 1_f64]);
        for i in 0..2 {
            assert_relative_eq!(deriv[i], 0.0, epsilon = f64::EPSILON);
        }

        let hessian = bukin_n6_hessian(&[-10_f64, 1_f64]);
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(hessian[i][j], 0.0, epsilon = f64::EPSILON);
            }
        }
    }

    proptest! {
        #[test]
        fn test_bukin_n6_derivative_finitediff(a in -15.0..-5.0, b in -3.0..3.0) {
            let param = [a, b];
            let derivative = bukin_n6_derivative(&param);
            let derivative_fd = Vec::from(param).central_diff(&|x| bukin_n6(&[x[0], x[1]]));
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
        fn test_bukin_n6_hessian_finitediff(a in -15.0..-5.0, b in -3.0..3.0) {
            let param = [a, b];
            let hessian = bukin_n6_hessian(&param);
            let hessian_fd = Vec::from(param).central_hessian(&|x| bukin_n6_derivative(&[x[0], x[1]]).to_vec());
            let n = hessian.len();
            // println!("1: {a}/{b} {hessian:?}");
            // println!("2: {a}/{b} {hessian_fd:?}");
            for i in 0..n {
                assert_eq!(hessian[i].len(), n);
                for j in 0..n {
                    assert_relative_eq!(
                        hessian[i][j],
                        hessian_fd[i][j],
                        epsilon = 1e-5,
                        max_relative = 1e-2,
                    );
                }
            }
        }
    }
}
