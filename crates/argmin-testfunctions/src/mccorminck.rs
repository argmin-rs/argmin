// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # McCorminck test function
//!
//! Defined as
//!
//! `f(x_1, x_2) = sin(x_1 + x_2) + (x_1 - x_2)^2 - 1.5*x_1 + 2.5*x_2 + 1`
//!
//! where `x_1 \in [-1.5, 4]` and `x_2 \in [-3, 4]`.
//!
//! The global minimum is at `f(x_1, x_2) = f(-0.54719, -1.54719) = -1.913228`.

use num::{Float, FromPrimitive};

/// McCorminck test function
///
/// Defined as
///
/// `f(x_1, x_2) = sin(x_1 + x_2) + (x_1 - x_2)^2 - 1.5*x_1 + 2.5*x_2 + 1`
///
/// where `x_1 \in [-1.5, 4]` and `x_2 \in [-3, 4]`.
///
/// The global minimum is at `f(x_1, x_2) = f(-0.54719, -1.54719) = -1.913228`.
pub fn mccorminck<T>(param: &[T; 2]) -> T
where
    T: Float + FromPrimitive,
{
    let [x1, x2] = *param;
    (x1 + x2).sin() + (x1 - x2).powi(2) - T::from_f64(1.5).unwrap() * x1
        + T::from_f64(2.5).unwrap() * x2
        + T::from_f64(1.0).unwrap()
}

/// Derivative of McCorminck test function
pub fn mccorminck_derivative<T>(param: &[T; 2]) -> [T; 2]
where
    T: Float + FromPrimitive,
{
    let [x1, x2] = *param;

    let n2 = T::from_f64(2.0).unwrap();
    let n3 = T::from_f64(3.0).unwrap();
    let n5 = T::from_f64(5.0).unwrap();

    [
        (x1 + x2).cos() + n2 * (x1 - x2) - n3 / n2,
        (x1 + x2).cos() - n2 * (x1 - x2) + n5 / n2,
    ]
}

/// Hessian of McCorminck test function
pub fn mccorminck_hessian<T>(param: &[T; 2]) -> [[T; 2]; 2]
where
    T: Float + FromPrimitive,
{
    let [x1, x2] = *param;

    let n2 = T::from_f64(2.0).unwrap();

    let a = (x1 + x2).sin();

    let diag = n2 - a;
    let offdiag = -n2 - a;

    [[diag, offdiag], [offdiag, diag]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use finitediff::FiniteDiff;
    use proptest::prelude::*;

    #[test]
    fn test_mccorminck_optimum() {
        assert_relative_eq!(
            mccorminck(&[-0.54719_f32, -1.54719_f32]),
            -1.9132228,
            epsilon = std::f32::EPSILON
        );
        assert_relative_eq!(
            mccorminck(&[-0.54719_f64, -1.54719_f64]),
            -1.9132229544882274,
            epsilon = std::f32::EPSILON.into()
        );

        let deriv = mccorminck_derivative(&[-0.54719_f64, -1.54719_f64]);
        println!("1: {deriv:?}");
        for i in 0..2 {
            assert_relative_eq!(deriv[i], 0.0, epsilon = 1e-4);
        }
    }

    proptest! {
        #[test]
        fn test_mccorminck_derivative_finitediff(a in -1.5..4.0, b in -3.0..4.0) {
            let param = [a, b];
            let derivative = mccorminck_derivative(&param);
            let derivative_fd = Vec::from(param).central_diff(&|x| mccorminck(&[x[0], x[1]]));
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
        fn test_mccorminck_hessian_finitediff(a in -1.5..4.0, b in -3.0..4.0) {
            let param = [a, b];
            let hessian = mccorminck_hessian(&param);
            let hessian_fd =
                Vec::from(param).central_hessian(&|x| mccorminck_derivative(&[x[0], x[1]]).to_vec());
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
