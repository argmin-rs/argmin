// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Holder table test function
//!
//! Defined as
//!
//! `f(x_1, x_2) = -abs(sin(x_1)*cos(x_2)*exp(abs(1- sqrt(x_1^2+x_2^2)/pi)))`
//!
//! where `x_i \in [-10, 10]`.
//!
//! The global minima are at
//!  * `f(x_1, x_2) = f(8.05502, 9.66459) = -19.2085`.
//!  * `f(x_1, x_2) = f(8.05502, -9.66459) = -19.2085`.
//!  * `f(x_1, x_2) = f(-8.05502, 9.66459) = -19.2085`.
//!  * `f(x_1, x_2) = f(-8.05502, -9.66459) = -19.2085`.

use num::{Float, FromPrimitive};
use std::f64::consts::PI;

/// Holder table test function
///
/// Defined as
///
/// `f(x_1, x_2) = -abs(sin(x_1)*cos(x_2)*exp(abs(1- sqrt(x_1^2+x_2^2)/pi)))`
///
/// where `x_i \in [-10, 10]`.
///
/// The global minima are at
///  * `f(x_1, x_2) = f(8.05502, 9.66459) = -19.2085`.
///  * `f(x_1, x_2) = f(8.05502, -9.66459) = -19.2085`.
///  * `f(x_1, x_2) = f(-8.05502, 9.66459) = -19.2085`.
///  * `f(x_1, x_2) = f(-8.05502, -9.66459) = -19.2085`.
pub fn holder_table<T>(param: &[T; 2]) -> T
where
    T: Float + FromPrimitive,
{
    let [x1, x2] = *param;
    let pi = T::from_f64(PI).unwrap();
    let n1 = T::from_f64(1.0).unwrap();
    -(x1.sin() * x2.cos() * (n1 - (x1.powi(2) + x2.powi(2)).sqrt() / pi).abs().exp()).abs()
}

/// Derivative of the Holder table test function
///
/// This function has a discontinuity at `sqrt(x_1^2+x_2^2) = PI`, and hence can return `NaN`.
pub fn holder_table_derivative<T>(param: &[T; 2]) -> [T; 2]
where
    T: Float + FromPrimitive,
{
    let [x1, x2] = *param;

    let pi = T::from_f64(PI).unwrap();
    let n0 = T::from_f64(0.0).unwrap();
    let n1 = T::from_f64(1.0).unwrap();
    let eps = T::epsilon();

    let x1sin = x1.sin();
    let x2sin = x2.sin();
    let x1cos = x1.cos();
    let x2cos = x2.cos();
    let x1sinabs = x1sin.abs();
    let x2cosabs = x2cos.abs();
    let a = (x1.powi(2) + x2.powi(2)).sqrt();
    let b = a / pi - n1;
    let c = b.abs();
    let d = c.exp();

    if a <= eps {
        [n0, n0]
    } else {
        [
            -(x1 * x1sinabs * x2cosabs * b * d) / (pi * a * c)
                - if x1sinabs <= eps {
                    n0
                } else {
                    (x1sin * x1cos * x2cosabs * d) / (x1sinabs)
                },
            -(x2 * x1sinabs * x2cosabs * b * d) / (pi * a * c)
                + if x2cosabs <= eps {
                    n0
                } else {
                    (x2sin * x2cos * x1sinabs * d) / (x2cosabs)
                },
        ]
    }
}

/// Hessian of the Holder table test function
///
/// This function has a discontinuity at `sqrt(x_1^2+x_2^2) = PI`, and hence can return `NaN`.
pub fn holder_table_hessian<T>(param: &[T; 2]) -> [[T; 2]; 2]
where
    T: Float + FromPrimitive,
{
    let [x1, x2] = *param;

    let pi = T::from_f64(PI).unwrap();
    let n0 = T::from_f64(0.0).unwrap();
    let n1 = T::from_f64(1.0).unwrap();
    let n2 = T::from_f64(2.0).unwrap();
    let eps = T::epsilon();

    let x1sin = x1.sin();
    let x2sin = x2.sin();
    let x1cos = x1.cos();
    let x2cos = x2.cos();
    let x1sinabs = x1sin.abs();
    let x2cosabs = x2cos.abs();
    let a = (x1.powi(2) + x2.powi(2)).sqrt();
    let b = a / pi - n1;
    let c = b.abs();
    let d = c.exp();

    let d1 = (x1sinabs * x2cosabs * d)
        * (if a <= eps {
            n0
        } else {
            -b / (pi * a * c) + (x1.powi(2) * b) / (pi * a.powi(3) * c)
                - (x1.powi(2)) / (pi.powi(2) * (x1.powi(2) + x2.powi(2)))
        } + n1)
        - if x1sinabs <= eps || a <= eps {
            n0
        } else {
            (n2 * x1 * x1sin * x1cos * x2cosabs * b * d) / (pi * a * c * x1sinabs)
        };

    let d2 = (x1sinabs * x2cosabs * d)
        * (if a <= eps {
            n0
        } else {
            -b / (pi * a * c) + (x2.powi(2) * b) / (pi * a.powi(3) * c)
                - (x2.powi(2)) / (pi.powi(2) * (x1.powi(2) + x2.powi(2)))
        } + n1)
        + if x2cosabs <= eps || a <= eps {
            n0
        } else {
            (n2 * x2 * x2sin * x2cos * x1sinabs * b * d) / (pi * a * c * x2cosabs)
        };

    let offdiag = if a <= eps || x1sinabs <= eps || x2cosabs <= eps {
        n0
    } else {
        d * (-(x2 * x1sin * x1cos * x2cosabs * b) / (pi * x1sinabs * a * c)
            + (x1 * x2 * x1sinabs * x2cosabs * b) / (pi * a.powi(3) * c)
            - (x1 * x2 * x1sinabs * x2cosabs) / (pi.powi(2) * (x1.powi(2) + x2.powi(2)))
            + (x1 * x1sinabs * x2sin * x2cos * b) / (pi * a * c * x2cosabs)
            + (x1sin * x1cos * x2sin * x2cos) / (x1sinabs * x2cosabs))
    };

    [[d1, offdiag], [offdiag, d2]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use finitediff::FiniteDiff;
    use proptest::prelude::*;
    use std::f32;

    #[test]
    fn test_holder_table_optimum() {
        assert_relative_eq!(
            holder_table(&[8.05502_f32, 9.66459_f32]),
            -19.2085,
            epsilon = f32::EPSILON
        );
        assert_relative_eq!(
            holder_table(&[-8.05502_f32, 9.66459_f32]),
            -19.2085,
            epsilon = f32::EPSILON
        );
        assert_relative_eq!(
            holder_table(&[8.05502_f32, -9.66459_f32]),
            -19.2085,
            epsilon = f32::EPSILON
        );
        assert_relative_eq!(
            holder_table(&[-8.05502_f32, -9.66459_f32]),
            -19.2085,
            epsilon = f32::EPSILON
        );

        for p in [
            [8.05502, 9.66459],
            [8.05502, -9.66459],
            [-8.05502, 9.66459],
            [-8.05502, -9.66459],
            [0.0, 0.0],
        ] {
            let deriv = holder_table_derivative(&p);
            for i in 0..2 {
                assert_relative_eq!(deriv[i], 0.0, epsilon = 1e-4);
            }
        }
    }

    proptest! {
        #[test]
        fn test_holder_table_derivative_finitediff(a in -10.0..10.0, b in -10.0..10.0) {
            let param = [a, b];
            let derivative = holder_table_derivative(&param);
            let derivative_fd = Vec::from(param).central_diff(&|x| holder_table(&[x[0], x[1]]));
            // println!("1: {derivative:?} at {a}/{b}");
            // println!("2: {derivative_fd:?} at {a}/{b}");
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
        fn test_holder_table_hessian_finitediff(a in -10.0..10.0, b in -10.0..10.0) {
            let param = [a, b];
            let hessian = holder_table_hessian(&param);
            let hessian_fd =
                Vec::from(param).forward_hessian(&|x| holder_table_derivative(&[x[0], x[1]]).to_vec());
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
