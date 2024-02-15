// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Easom test function
//!
//! Defined as
//!
//! `f(x_1, x_2) = - cos(x_1) * cos(x_2) * exp(-(x_1 - pi)^2 - (x_2 - pi)^2)`
//!
//! where `x_i \in [-100, 100]`.
//!
//! The global minimum is at `f(x_1, x_2) = f(pi, pi) = -1`.

use num::{Float, FromPrimitive};
use std::f64::consts::PI;

/// Easom test function
///
/// Defined as
///
/// `f(x_1, x_2) = - cos(x_1) * cos(x_2) * exp(-(x_1 - pi)^2 - (x_2 - pi)^2)`
///
/// where `x_i \in [-100, 100]`.
///
/// The global minimum is at `f(x_1, x_2) = f(pi, pi) = -1`.
pub fn easom<T>(param: &[T; 2]) -> T
where
    T: Float + FromPrimitive,
{
    let [x1, x2] = *param;
    let pi = T::from_f64(PI).unwrap();
    -x1.cos() * x2.cos() * (-(x1 - pi).powi(2) - (x2 - pi).powi(2)).exp()
}

/// Derivative of Easom test function
pub fn easom_derivative<T>(param: &[T; 2]) -> [T; 2]
where
    T: Float + FromPrimitive,
{
    let [x1, x2] = *param;

    let pi = T::from_f64(PI).unwrap();
    let n2 = T::from_f64(2.0).unwrap();

    let factor = (-(x1 - pi).powi(2) - (x2 - pi).powi(2)).exp();

    [
        factor * x2.cos() * (x1.sin() + n2 * x1 * x1.cos() - n2 * pi * x1.cos()),
        factor * x1.cos() * (x2.sin() + n2 * x2 * x2.cos() - n2 * pi * x2.cos()),
    ]
}

/// Hessian of Easom test function
pub fn easom_hessian<T>(param: &[T; 2]) -> [[T; 2]; 2]
where
    T: Float + FromPrimitive,
{
    let [x1, x2] = *param;

    let pi = T::from_f64(PI).unwrap();
    let n2 = T::from_f64(2.0).unwrap();
    let n4 = T::from_f64(4.0).unwrap();
    let n3 = T::from_f64(3.0).unwrap();
    let n8 = T::from_f64(8.0).unwrap();

    let x1cos = x1.cos();
    let x1sin = x1.sin();
    let x2cos = x2.cos();
    let x2sin = x2.sin();
    let factor = (-(x1 - pi).powi(2) - (x2 - pi).powi(2)).exp();
    let offdiag = factor * (x1sin + n2 * (x1 - pi) * x1cos) * (n2 * (pi - x2) * x2cos - x2sin);

    [
        [
            factor
                * x2cos
                * (n4 * (pi - x1) * x1sin
                    + (-n4 * x1.powi(2) + n8 * pi * x1 - n4 * pi.powi(2) + n3) * x1cos),
            offdiag,
        ],
        [
            offdiag,
            factor
                * x1cos
                * (n4 * (pi - x2) * x2sin
                    + (-n4 * x2.powi(2) + n8 * pi * x2 - n4 * pi.powi(2) + n3) * x2cos),
        ],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use finitediff::FiniteDiff;
    use proptest::prelude::*;
    use std::{f32, f32::consts::PI as PI32, f64, f64::consts::PI as PI64};

    #[test]
    fn test_easom_optimum() {
        assert_relative_eq!(easom(&[PI32, PI32]), -1.0_f32, epsilon = f32::EPSILON);
        assert_relative_eq!(easom(&[PI64, PI64]), -1.0_f64, epsilon = f64::EPSILON);

        let deriv = easom_derivative(&[PI64, PI64]);
        for i in 0..2 {
            assert_relative_eq!(deriv[i], 0.0_f64, epsilon = f64::EPSILON);
        }
    }

    proptest! {
        #[test]
        fn test_easom_derivative_finitediff(a in -100.0..100.0, b in -100.0..100.0) {
            let param = [a, b];
            let derivative = easom_derivative(&param);
            let derivative_fd = Vec::from(param).central_diff(&|x| easom(&[x[0], x[1]]));
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
        fn test_easom_hessian_finitediff(a in -100.0..100.0, b in -100.0..100.0) {
            let param = [a, b];
            let hessian = easom_hessian(&param);
            let hessian_fd =
                Vec::from(param).forward_hessian(&|x| easom_derivative(&[x[0], x[1]]).to_vec());
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
