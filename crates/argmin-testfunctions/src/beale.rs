// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Beale test function
//!
//! Defined as
//!
//! `f(x_1, x_2) = (1.5 - x_1 + x_1 * x_2)^2 + (2.25 - x_1 + x_1 * x_2^2)^2 +
//!                (2.625 - x_1 + x1 * x_2^3)^2`
//!
//! where `x_i \in [-4.5, 4.5]`.
//!
//! The global minimum is at `f(x_1, x_2) = f(3, 0.5) = 0`.

use num::{Float, FromPrimitive};

/// Beale test function
///
/// Defined as
///
/// `f(x_1, x_2) = (1.5 - x_1 + x_1 * x_2)^2 + (2.25 - x_1 + x_1 * x_2^2)^2 +
///                (2.625 - x_1 + x_1 * x_2^3)^2`
///
/// where `x_i \in [-4.5, 4.5]`.
///
/// The global minimum is at `f(x_1, x_2) = f(3, 0.5) = 0`.
pub fn beale<T>(param: &[T; 2]) -> T
where
    T: Float + FromPrimitive,
{
    let [x1, x2] = *param;
    (T::from_f64(1.5).unwrap() - x1 + x1 * x2).powi(2)
        + (T::from_f64(2.25).unwrap() - x1 + x1 * (x2.powi(2))).powi(2)
        + (T::from_f64(2.625).unwrap() - x1 + x1 * (x2.powi(3))).powi(2)
}

/// Derivative of Beale test function
pub fn beale_derivative<T>(param: &[T; 2]) -> [T; 2]
where
    T: Float + FromPrimitive,
{
    let n0_5 = T::from_f64(0.5).unwrap();
    let n1 = T::from_f64(1.0).unwrap();
    let n1_5 = T::from_f64(1.5).unwrap();
    let n2 = T::from_f64(2.0).unwrap();
    let n2_625 = T::from_f64(2.625).unwrap();
    let n3 = T::from_f64(3.0).unwrap();
    let n4_5 = T::from_f64(4.5).unwrap();
    let n5_25 = T::from_f64(5.25).unwrap();
    let n6 = T::from_f64(6.0).unwrap();
    let n12_75 = T::from_f64(12.75).unwrap();

    let [x1, x2] = *param;

    let x2p2 = x2.powi(2);
    let x2p3 = x2.powi(3);
    let x2p4 = x2.powi(4);
    let x2p5 = x2.powi(5);
    let x2p6 = x2.powi(6);

    [
        n5_25 * x2p3
            + n4_5 * x2p2
            + n3 * x2
            + n2 * x1 * (x2p6 + x2p4 - n2 * x2p3 - x2p2 - n2 * x2 + n3)
            - n12_75,
        n6 * x1
            * (n2_625 * x2p2
                + n1_5 * x2
                + x1 * (x2p5 + (n2 / n3) * x2p3 - x2p2 - (n1 / n3) * (x2 + n1))
                + n0_5),
    ]
}

/// Derivative of Beale test function
pub fn beale_hessian<T>(param: &[T; 2]) -> [[T; 2]; 2]
where
    T: Float + FromPrimitive,
{
    let n2 = T::from_f64(2.0).unwrap();
    let n3 = T::from_f64(3.0).unwrap();
    let n4 = T::from_f64(4.0).unwrap();
    let n8 = T::from_f64(8.0).unwrap();
    let n9 = T::from_f64(9.0).unwrap();
    let n12 = T::from_f64(12.0).unwrap();
    let n15_75 = T::from_f64(15.75).unwrap();
    let n30 = T::from_f64(30.0).unwrap();
    let n31_5 = T::from_f64(31.5).unwrap();

    let [x1, x2] = *param;

    let x1p2 = x1.powi(2);
    let x2p2 = x2.powi(2);
    let x2p3 = x2.powi(3);
    let x2p4 = x2.powi(4);
    let x2p5 = x2.powi(5);
    let x2p6 = x2.powi(6);

    let offdiag = n12 * x1 * x2p5 + n8 * x1 * x2p3 - n12 * x1 * x2p2 + n15_75 * x2p2 - n4 * x1 * x2
        + n9 * x2
        - n4 * x1
        + n3;

    [
        [
            n2 * (x2p6 + x2p4 - n2 * x2p3 - x2p2 - n2 * x2 + n3),
            offdiag,
        ],
        [
            offdiag,
            n30 * x1p2 * x2p4 + n12 * x1p2 * x2p2 - n12 * x1p2 * x2 + n31_5 * x1 * x2 - n2 * x1p2
                + n9 * x1,
        ],
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
    fn test_beale_optimum() {
        assert_relative_eq!(beale(&[3.0_f32, 0.5_f32]), 0.0, epsilon = f32::EPSILON);
        assert_relative_eq!(beale(&[3.0_f64, 0.5_f64]), 0.0, epsilon = f64::EPSILON);
    }

    proptest! {
        #[test]
        fn test_beale_derivative(a in -4.5..4.5, b in -4.5..4.5) {
            let param = [a, b];
            let derivative = beale_derivative(&param);
            let derivative_fd = Vec::from(param).central_diff(&|x| beale(&[x[0], x[1]]));
            for i in 0..derivative.len() {
                assert_relative_eq!(derivative[i], derivative_fd[i], epsilon = 1e-2);
            }
        }
    }

    proptest! {
        #[test]
        fn test_beale_hessian_finitediff(a in -4.5..4.5, b in -4.5..4.5) {
            let param = [a, b];
            let hessian = beale_hessian(&param);
            let hessian_fd =
                Vec::from(param).forward_hessian(&|x| beale_derivative(&[x[0], x[1]]).to_vec());
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
