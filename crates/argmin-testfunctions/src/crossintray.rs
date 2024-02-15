// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Cross-in-tray test function
//!
//! Defined as
//!
//! `f(x_1, x_2) = -0.0001 * ( | sin(x_1)*sin(x_2)*exp(| 100 -
//!                                                      \sqrt{x_1^2+x_2^2} / pi |) | + 1)^0.1`
//!
//! where `x_i \in [-10, 10]`.
//!
//! The global minima are at
//!  * `f(x_1, x_2) = f(1.34941, 1.34941) = -2.06261`.
//!  * `f(x_1, x_2) = f(1.34941, -1.34941) = -2.06261`.
//!  * `f(x_1, x_2) = f(-1.34941, 1.34941) = -2.06261`.
//!  * `f(x_1, x_2) = f(-1.34941, -1.34941) = -2.06261`.

use std::f64::consts::PI;

use num::{Float, FromPrimitive};

/// Cross-in-tray test function
///
/// Defined as
///
/// `f(x_1, x_2) = -0.0001 * ( | sin(x_1)*sin(x_2)*exp(| 100 -
///                                                      \sqrt{x_1^2+x_2^2} / pi |) | + 1)^0.1`
///
/// where `x_i \in [-10, 10]`.
///
/// The global minima are at
///  * `f(x_1, x_2) = f(1.34941, 1.34941) = -2.06261`.
///  * `f(x_1, x_2) = f(1.34941, -1.34941) = -2.06261`.
///  * `f(x_1, x_2) = f(-1.34941, 1.34941) = -2.06261`.
///  * `f(x_1, x_2) = f(-1.34941, -1.34941) = -2.06261`.
///
/// Note: Even if the input parameters are f32, internal computations will be performed in f64.
pub fn cross_in_tray<T>(param: &[T; 2]) -> T
where
    T: Float + Into<f64> + FromPrimitive,
{
    let x1: f64 = param[0].into();
    let x2: f64 = param[1].into();
    T::from_f64(
        -0.0001
            * ((x1.sin() * x2.sin() * (100.0 - (x1.powi(2) + x2.powi(2)).sqrt() / PI).abs().exp())
                .abs()
                + 1.0)
                .powf(0.1),
    )
    .unwrap()
}

/// Derivative of Cross-in-tray test function
///
/// Note: Even if the input parameters are f32, internal computations will be performed in f64.
pub fn cross_in_tray_derivative<T>(param: &[T; 2]) -> [T; 2]
where
    T: Float + Into<f64> + FromPrimitive,
{
    let x1: f64 = param[0].into();
    let x2: f64 = param[1].into();

    let a = (x1.powi(2) + x2.powi(2)).sqrt();
    let b = a / PI - 100.0;
    let c = b.abs().exp();
    let x2sin = x2.sin();
    let x2sinabs = x2sin.abs();
    let x1sin = x1.sin();
    let x1sinabs = x1sin.abs();
    let x1cos = x1.cos();
    let x2cos = x2.cos();
    let denom = PI * a * b.abs();

    [
        T::from_f64({
            -1e-5
                * (if denom.abs() <= f64::EPSILON {
                    0.0
                } else {
                    (x2sinabs * x1 * b * c * x1sinabs) / denom
                } + if x1sinabs <= f64::EPSILON {
                    0.0
                } else {
                    (x2sinabs * x1sin * x1cos * c) / x1sinabs
                })
                / (x2sinabs * x1sinabs * c + 1.0).powf(0.9)
        })
        .unwrap(),
        T::from_f64({
            -1e-5
                * (if denom.abs() <= f64::EPSILON {
                    0.0
                } else {
                    (x2sinabs * x2 * b * c * x1sinabs) / denom
                } + if x2sinabs <= f64::EPSILON {
                    0.0
                } else {
                    (x1sinabs * x2sin * x2cos * c) / x2sinabs
                })
                / (x2sinabs * x1sinabs * c + 1.0).powf(0.9)
        })
        .unwrap(),
    ]
}

/// Hessian of Cross-in-tray test function
///
/// This function may return NaN or INF.
///
/// Note: Even if the input parameters are f32, internal computations will be performed in f64.
pub fn cross_in_tray_hessian<T>(param: &[T; 2]) -> [[T; 2]; 2]
where
    T: Float + Into<f64> + FromPrimitive,
{
    let x1: f64 = param[0].into();
    let x2: f64 = param[1].into();

    let a = (x1.powi(2) + x2.powi(2)).sqrt();
    let b = a / PI - 100.0;
    let c = b.abs().exp();
    let x2sin = x2.sin();
    let x2sinabs = x2sin.abs();
    let x1sin = x1.sin();
    let x1sinabs = x1sin.abs();
    let x1cos = x1.cos();
    let x2cos = x2.cos();
    let denom = PI * a * b.abs();
    let d = x1sinabs * x2sinabs * b * c / denom;

    let h1 = T::from_f64({
        9.0 * 1e-6 * (d * x1 + (x1sin * x1cos * x2sinabs * c) / x1sinabs).powi(2)
            / (x1sinabs * x2sinabs * c + 1.0).powf(1.9)
            - 1e-5
                * (d - x1.powi(2)
                    * ((x1sinabs * x2sinabs * b * c) / (PI * a.powi(3) * b.abs())
                        - (x1sinabs * x2sinabs * c) / (PI.powi(2) * (x1.powi(2) + x2.powi(2))))
                    - x2sinabs * x1sinabs * c
                    + (2.0 * x1sin * x1cos * x2sinabs * x1 * b * c) / (denom * x1sinabs))
                / (x1sinabs * x2sinabs * c + 1.0).powf(0.9)
    })
    .unwrap();

    let h2 = T::from_f64({
        9.0 * 1e-6 * (d * x2 + (x2sin * x2cos * x1sinabs * c) / x2sinabs).powi(2)
            / (x1sinabs * x2sinabs * c + 1.0).powf(1.9)
            - 1e-5
                * (d - x2.powi(2)
                    * ((x1sinabs * x2sinabs * b * c) / (PI * a.powi(3) * b.abs())
                        - (x1sinabs * x2sinabs * c) / (PI.powi(2) * (x1.powi(2) + x2.powi(2))))
                    - x2sinabs * x1sinabs * c
                    + (2.0 * x2sin * x2cos * x1sinabs * x2 * b * c) / (denom * x2sinabs))
                / (x1sinabs * x2sinabs * c + 1.0).powf(0.9)
    })
    .unwrap();

    let offdiag = T::from_f64({
        (9.0 * 1e-6
            * (x1 * d + (x1sin * x1cos * x2sinabs * c) / x1sinabs)
            * (x2 * d + (x2sin * x2cos * x1sinabs * c) / x2sinabs))
            / (x1sinabs * x2sinabs * c + 1.0).powf(1.9)
            - ((x2 * x1sin * x1cos * x2sinabs * b * c) / (PI * x1sinabs * a * b.abs())
                - (x1 * x2 * x1sinabs * x2sinabs * b * c) / (PI * a.powi(3) * b.abs())
                + (x1 * x2 * x1sinabs * x2sinabs * c) / (PI.powi(2) * (x1.powi(2) + x2.powi(2)))
                + (x1 * x2sin * x2cos * x1sinabs * b * c) / (PI * a * b.abs() * x2sinabs)
                + (x1sin * x1cos * x2sin * x2cos * c) / (x1sinabs * x2sinabs))
                / (100_000.0 * (x1sinabs * x2sinabs * c + 1.0).powf(0.9))
    })
    .unwrap();

    [[h1, offdiag], [offdiag, h2]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use finitediff::FiniteDiff;
    use proptest::prelude::*;
    use std::f32;

    #[test]
    fn test_cross_in_tray_optimum() {
        // This isnt exactly a great way to test this. The function can only be computed with the
        // use of f64; however, I only have the minimum points available in f32, which is why I use
        // the f32 EPSILONs.
        assert_relative_eq!(
            cross_in_tray(&[1.34941_f64, 1.34941_f64]),
            -2.062611870,
            epsilon = f32::EPSILON.into()
        );
        assert_relative_eq!(
            cross_in_tray(&[1.34941_f64, -1.34941_f64]),
            -2.062611870,
            epsilon = f32::EPSILON.into()
        );
        assert_relative_eq!(
            cross_in_tray(&[-1.34941_f64, 1.34941_f64]),
            -2.062611870,
            epsilon = f32::EPSILON.into()
        );
        assert_relative_eq!(
            cross_in_tray(&[-1.34941_f64, -1.34941_f64]),
            -2.062611870,
            epsilon = f32::EPSILON.into()
        );
        assert_relative_eq!(
            cross_in_tray(&[1.34941_f32, 1.34941_f32]),
            -2.062611870,
            epsilon = f32::EPSILON.into()
        );
        assert_relative_eq!(
            cross_in_tray(&[1.34941_f32, -1.34941_f32]),
            -2.062611870,
            epsilon = f32::EPSILON.into()
        );
        assert_relative_eq!(
            cross_in_tray(&[-1.34941_f32, 1.34941_f32]),
            -2.062611870,
            epsilon = f32::EPSILON.into()
        );
        assert_relative_eq!(
            cross_in_tray(&[-1.34941_f32, -1.34941_f32]),
            -2.062611870,
            epsilon = f32::EPSILON.into()
        );

        for p in [
            [1.34941_f64, 1.34941_f64],
            [1.34941_f64, -1.34941_f64],
            [-1.34941_f64, 1.34941_f64],
            [-1.34941_f64, -1.34941_f64],
            [0.0_f64, 0.0_f64],
        ] {
            let deriv = cross_in_tray_derivative(&p);
            for i in 0..2 {
                assert_relative_eq!(deriv[i], 0.0, epsilon = 1e-6);
            }
        }
    }

    proptest! {
        #[test]
        fn test_cross_in_tray_derivative_finitediff(a in -10.0..10.0, b in -10.0..10.0) {
            let param = [a, b];
            let derivative = cross_in_tray_derivative(&param);
            let derivative_fd = Vec::from(param).central_diff(&|x| cross_in_tray(&[x[0], x[1]]));
            // println!("1: {derivative:?} at {a}/{b}");
            // println!("2: {derivative_fd:?} at {a}/{b}");
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
        fn test_cross_in_tray_hessian_finitediff(a in -10.0..10.0, b in -10.0..10.0) {
            let param = [a, b];
            let hessian = cross_in_tray_hessian(&param);
            let hessian_fd =
                Vec::from(param).central_hessian(&|x| cross_in_tray_derivative(&[x[0], x[1]]).to_vec());
            let n = hessian.len();
            // println!("1: {hessian:?} at {a}/{b}");
            // println!("2: {hessian_fd:?} at {a}/{b}");
            for i in 0..n {
                assert_eq!(hessian[i].len(), n);
                for j in 0..n {
                    if hessian[i][j].is_finite() {
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
