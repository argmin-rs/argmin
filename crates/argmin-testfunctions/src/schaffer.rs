// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Schaffer test function No. 2
//!
//! Defined as
//!
//! `f(x_1, x_2) = 0.5 + (sin^2(x_1^2 - x_2^2) - 0.5) / (1 + 0.001*(x_1^2 + x_2^2))^2`
//!
//! where `x_i \in [-100, 100]`.
//!
//! The global minimum is at `f(x_1, x_2) = f(0, 0) = 0`.
//!
//! # Schaffer test function No. 4
//!
//! Defined as
//!
//! `f(x_1, x_2) = 0.5 + (cos(sin(abs(x_1^2 - x_2^2)))^2 - 0.5) / (1 + 0.001*(x_1^2 + x_2^2))^2`
//!
//! where `x_i \in [-100, 100]`.
//!
//! The global minimum is at `f(x_1, x_2) = f(0, 1.25313) = 0.291992`.

use num::{Float, FromPrimitive};

/// Schaffer test function No. 2
///
/// Defined as
///
/// `f(x_1, x_2) = 0.5 + (sin^2(x_1^2 - x_2^2) - 0.5) / (1 + 0.001*(x_1^2 + x_2^2))^2`
///
/// where `x_i \in [-100, 100]`.
///
/// The global minimum is at `f(x_1, x_2) = f(0, 0) = 0`.
pub fn schaffer_n2<T>(param: &[T; 2]) -> T
where
    T: Float + FromPrimitive,
{
    let [x1, x2] = *param;

    let n0_001 = T::from_f64(0.001).unwrap();
    let n0_5 = T::from_f64(0.5).unwrap();
    let n1 = T::from_f64(1.0).unwrap();

    n0_5 + ((x1.powi(2) - x2.powi(2)).sin().powi(2) - n0_5)
        / (n1 + n0_001 * (x1.powi(2) + x2.powi(2))).powi(2)
}

/// Derivative of Schaffer test function No. 2
pub fn schaffer_n2_derivative<T>(param: &[T; 2]) -> [T; 2]
where
    T: Float + FromPrimitive,
{
    let [x1, x2] = *param;

    let n0_001 = T::from_f64(0.001).unwrap();
    let n0_004 = T::from_f64(0.004).unwrap();
    let n0_5 = T::from_f64(0.5).unwrap();
    let n1 = T::from_f64(1.0).unwrap();
    let n4 = T::from_f64(4.0).unwrap();

    let x1spx2s = x1.powi(2) + x2.powi(2);
    let x1smx2s = x1.powi(2) - x2.powi(2);
    let x2smx1s = x2.powi(2) - x1.powi(2);
    let tmp = n0_001 * x1spx2s + n1;
    let denom2 = tmp.powi(2);
    let denom3 = tmp.powi(3);
    let a = x1smx2s.sin() * x1smx2s.cos();
    let a2 = x2smx1s.sin() * x2smx1s.cos();
    let b = n0_004 * (x1smx2s.sin().powi(2) - n0_5);

    [
        (n4 * x1 * a) / denom2 - (x1 * b) / denom3,
        (n4 * x2 * a2) / denom2 - (x2 * b) / denom3,
    ]
}

/// Hessian of Schaffer test function No. 2
pub fn schaffer_n2_hessian<T>(param: &[T; 2]) -> [[T; 2]; 2]
where
    T: Float + FromPrimitive,
{
    let [x1, x2] = *param;

    let n0_001 = T::from_f64(0.001).unwrap();
    let n0_004 = T::from_f64(0.004).unwrap();
    let n0_006 = T::from_f64(0.006).unwrap();
    let n0_032 = T::from_f64(0.032).unwrap();
    let n0_5 = T::from_f64(0.5).unwrap();
    let n1 = T::from_f64(1.0).unwrap();
    let n4 = T::from_f64(4.0).unwrap();
    let n8 = T::from_f64(8.0).unwrap();

    let x1s = x1.powi(2);
    let x2s = x2.powi(2);
    let x1spx2s = x1s + x2s;
    let x1smx2s = x1s - x2s;
    let x2smx1s = x2s - x1s;
    let x1smx2ssin2 = x1smx2s.sin().powi(2);
    let x2smx1ssin2 = x2smx1s.sin().powi(2);
    let x1smx2scos2 = x1smx2s.cos().powi(2);
    let x2smx1scos2 = x2smx1s.cos().powi(2);
    let tmp = n0_001 * x1spx2s + n1;
    let denom2 = tmp.powi(2);
    let denom3 = tmp.powi(3);
    let denom4 = tmp.powi(4);
    let a = x1smx2s.sin() * x1smx2s.cos();
    let a2 = x2smx1s.sin() * x2smx1s.cos();
    let b = n0_004 * (x1smx2ssin2 - n0_5);

    let offdiag =
        (n8 * x1 * x2 * (x1smx2ssin2 - x1smx2scos2)) / denom2 + (n0_006 * x1 * x2 * b) / denom4;

    [
        [
            (-(n8 * x1s * x1smx2ssin2) + (n4 * a) + (n8 * x1s * x1smx2scos2)) / denom2
                - (b + (n0_032 * x1s * a)) / denom3
                + (n0_006 * x1s * b) / denom4,
            offdiag,
        ],
        [
            offdiag,
            (-(n8 * x2s * x2smx1ssin2) + (n4 * a2) + (n8 * x2s * x2smx1scos2)) / denom2
                - (b + (n0_032 * x2s * a2)) / denom3
                + (n0_006 * x2s * b) / denom4,
        ],
    ]
}

/// Schaffer test function No. 4
///
/// Defined as
///
/// `f(x_1, x_2) = 0.5 + (cos(sin(abs(x_1^2 - x_2^2)))^2 - 0.5) / (1 + 0.001*(x_1^2 + x_2^2))^2`
///
/// where `x_i \in [-100, 100]`.
///
/// The global minimum is at `f(x_1, x_2) = f(0, 1.25313) = 0.291992`.
pub fn schaffer_n4<T>(param: &[T; 2]) -> T
where
    T: Float + FromPrimitive,
{
    let [x1, x2] = *param;
    let n05 = T::from_f64(0.5).unwrap();
    let n1 = T::from_f64(1.0).unwrap();
    let n0001 = T::from_f64(0.001).unwrap();
    n05 + ((x1.powi(2) - x2.powi(2)).abs().sin().cos().powi(2) - n05)
        / (n1 + n0001 * (x1.powi(2) + x2.powi(2))).powi(2)
}

/// Derivative of Schaffer test function No. 4
pub fn schaffer_n4_derivative<T>(param: &[T; 2]) -> [T; 2]
where
    T: Float + FromPrimitive,
{
    let [x1, x2] = *param;
    let n0_5 = T::from_f64(0.5).unwrap();
    let n1 = T::from_f64(1.0).unwrap();
    let n0_001 = T::from_f64(0.001).unwrap();
    let n0_004 = T::from_f64(0.004).unwrap();
    let n4 = T::from_f64(4.0).unwrap();

    let x1smx2s = x1.powi(2) - x2.powi(2);
    let x2smx1s = x2.powi(2) - x1.powi(2);
    let x1spx2s = x1.powi(2) + x2.powi(2);
    let x1smx2scos = x1smx2s.cos();
    let x2smx1scos = x2smx1s.cos();
    let x1smx2sabs = x1smx2s.abs();
    let x1smx2sabssin = x1smx2sabs.sin();
    let x2smx1sabssin = x1smx2sabs.sin();
    let x1smx2sabssincos = x1smx2sabssin.cos();
    let x1smx2sabssinsin = x1smx2sabssin.sin();
    let x1smx2sabssincos2 = x1smx2sabssin.cos().powi(2);
    let x2smx1sabssincos = x2smx1sabssin.cos();
    let x2smx1sabssinsin = x2smx1sabssin.sin();
    let x2smx1sabssincos2 = x2smx1sabssin.cos().powi(2);
    let denom_a = (n0_001 * x1spx2s + n1).powi(2);
    let denom_b = (n0_001 * x1spx2s + n1).powi(3);

    [
        -(n4 * x1 * x1smx2s * x1smx2scos * x1smx2sabssincos * x1smx2sabssinsin)
            / (denom_a * x1smx2sabs)
            - (n0_004 * x1 * (x1smx2sabssincos2 - n0_5)) / denom_b,
        -(n4 * x2 * x2smx1s * x2smx1scos * x2smx1sabssincos * x2smx1sabssinsin)
            / (denom_a * x1smx2sabs)
            - (n0_004 * x2 * (x2smx1sabssincos2 - n0_5)) / denom_b,
    ]
}

/// Hessian of Schaffer test function No. 4
pub fn schaffer_n4_hessian<T>(param: &[T; 2]) -> [[T; 2]; 2]
where
    T: Float + FromPrimitive,
{
    let [x1, x2] = *param;

    let eps = T::epsilon();
    let n0_5 = T::from_f64(0.5).unwrap();
    let n0 = T::from_f64(0.0).unwrap();
    let n1 = T::from_f64(1.0).unwrap();
    let n0_001 = T::from_f64(0.001).unwrap();
    let n0_004 = T::from_f64(0.004).unwrap();
    let n0_006 = T::from_f64(0.006).unwrap();
    let n0_016 = T::from_f64(0.016).unwrap();
    let n0_032 = T::from_f64(0.032).unwrap();
    let n4 = T::from_f64(4.0).unwrap();
    let n8 = T::from_f64(8.0).unwrap();

    let x1s = x1.powi(2);
    let x2s = x2.powi(2);
    let x1smx2s = x1s - x2s;
    let x2smx1s = x2s - x1s;
    let x1spx2s = x1s + x2s;
    let x1smx2scos = x1smx2s.cos();
    let x1smx2ssin = x1smx2s.sin();
    let x2smx1ssin = x2smx1s.sin();
    let x2smx1scos = x2smx1s.cos();
    let x1smx2scos2 = x1smx2scos.powi(2);
    let x2smx1scos2 = x2smx1scos.powi(2);
    let x1smx2sabs = x1smx2s.abs();
    let x1smx2sabssin = x1smx2sabs.sin();
    let x2smx1sabssin = x1smx2sabs.sin();
    let x1smx2sabssincos = x1smx2sabssin.cos();
    let x1smx2sabssinsin = x1smx2sabssin.sin();
    let x2smx1sabssinsin = x2smx1sabssin.sin();
    let x1smx2sabssinsin2 = x1smx2sabssinsin.powi(2);
    let x2smx1sabssinsin2 = x2smx1sabssinsin.powi(2);
    let x1smx2sabssincos2 = x1smx2sabssin.cos().powi(2);
    let x2smx1sabssincos = x2smx1sabssin.cos();
    let x2smx1sabssinsin = x2smx1sabssin.sin();
    let x2smx1sabssincos2 = x2smx1sabssin.cos().powi(2);
    let denom_a = (n0_001 * x1spx2s + n1).powi(2);
    let denom_b = (n0_001 * x1spx2s + n1).powi(3);
    let denom_c = (n0_001 * x1spx2s + n1).powi(4);

    if x1smx2sabs <= eps {
        [[n0, n0], [n0, n0]]
    } else {
        let a = (n8 * x1s * x1smx2scos2 * x1smx2sabssinsin2
            - n8 * x1s * x1smx2scos2 * x1smx2sabssincos2)
            / denom_a
            + ((n8 * x1s * x1smx2s * x1smx2ssin * x1smx2sabssincos * x1smx2sabssinsin)
                - (n4 * x1smx2s * x1smx2scos * x1smx2sabssincos * x1smx2sabssinsin))
                / (denom_a * x1smx2sabs)
            + (n0_032 * x1s * x1smx2s * x1smx2scos * x1smx2sabssincos * x1smx2sabssinsin)
                / (denom_b * x1smx2sabs)
            + (-n0_004 * (x1smx2sabssincos2 - n0_5)) / denom_b
            + (n0_006 * n0_004 * x1s * (x1smx2sabssincos2 - n0_5)) / denom_c;

        let b = (n8 * x2s * x2smx1scos2 * x2smx1sabssinsin2
            - n8 * x2s * x2smx1scos2 * x2smx1sabssincos2)
            / denom_a
            + ((n8 * x2s * x2smx1s * x2smx1ssin * x2smx1sabssincos * x2smx1sabssinsin)
                - (n4 * x2smx1s * x2smx1scos * x2smx1sabssincos * x2smx1sabssinsin))
                / (denom_a * x1smx2sabs)
            + (n0_032 * x2s * x2smx1s * x2smx1scos * x2smx1sabssincos * x2smx1sabssinsin)
                / (denom_b * x1smx2sabs)
            + (-n0_004 * (x2smx1sabssincos2 - n0_5)) / denom_b
            + (n0_006 * n0_004 * x2s * (x2smx1sabssincos2 - n0_5)) / denom_c;

        let offdiag = (n8 * x1 * x2 * x1smx2s * x2smx1scos2 * x2smx1sabssinsin2)
            / (x2smx1s * denom_a)
            + (n8 * x1 * x2 * x1smx2s * x2smx1ssin * x2smx1sabssincos * x2smx1sabssinsin)
                / (x1smx2sabs * denom_a)
            + (n8 * x1 * x2 * x1smx2s * x2smx1scos * x2smx1sabssincos * x2smx1sabssinsin)
                / (x2smx1s * denom_a * x1smx2sabs)
            + (n8 * x1 * x2 * x2smx1scos * x2smx1sabssincos * x2smx1sabssinsin)
                / (denom_a * x1smx2sabs)
            + (n0_016 * x1 * x2 * x2smx1s * x2smx1scos * x2smx1sabssincos * x2smx1sabssinsin)
                / (denom_b * x1smx2sabs)
            + (n0_016 * x1 * x2 * x1smx2s * x2smx1scos * x2smx1sabssincos * x2smx1sabssinsin)
                / (denom_b * x1smx2sabs)
            + (-n8 * x1 * x2 * x1smx2s * x2smx1scos2 * x2smx1sabssincos2) / (denom_a * x2smx1s)
            + (n0_006 * n0_004 * x1 * x2 * (x2smx1sabssincos2 - n0_5)) / denom_c;

        [[a, offdiag], [offdiag, b]]
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
    fn test_schaffer_n2_optimum() {
        assert_relative_eq!(schaffer_n2(&[0_f32, 0_f32]), 0.0, epsilon = f32::EPSILON);
        assert_relative_eq!(schaffer_n2(&[0_f64, 0_f64]), 0.0, epsilon = f64::EPSILON);

        let deriv = schaffer_n2_derivative(&[0.0, 0.0]);
        for i in 0..2 {
            assert_relative_eq!(deriv[i], 0.0, epsilon = f64::EPSILON);
        }
    }

    proptest! {
        #[test]
        fn test_schaffer_n2_derivative_finitediff(a in -10.0..10.0, b in -10.0..10.0) {
            // Note: prop testing range is smaller than range of function, simply because finitediff
            // has huge errors far away from the optimum.
            let param = [a, b];
            let derivative = schaffer_n2_derivative(&param);
            let derivative_fd = Vec::from(param).central_diff(&|x| schaffer_n2(&[x[0], x[1]]));
            // println!("1: {derivative:?} at {a}/{b}");
            // println!("2: {derivative_fd:?} at {a}/{b}");
            for i in 0..derivative.len() {
                assert_relative_eq!(
                    derivative[i],
                    derivative_fd[i],
                    epsilon = 1e-3,
                    max_relative = 1e-2
                );
            }
        }
    }

    proptest! {
        #[test]
        fn test_schaffer_n2_hessian_finitediff(a in -10.0..10.0, b in -10.0..10.0) {
            // Note: prop testing range is smaller than range of function, simply because finitediff
            // has huge errors far away from the optimum.
            let param = [a, b];
            let hessian = schaffer_n2_hessian(&param);
            let hessian_fd =
                Vec::from(param).forward_hessian(&|x| schaffer_n2_derivative(&[x[0], x[1]]).to_vec());
            let n = hessian.len();
            // println!("1: {hessian:?} at {a}/{b}");
            // println!("2: {hessian_fd:?} at {a}/{b}");
            for i in 0..n {
                assert_eq!(hessian[i].len(), n);
                for j in 0..n {
                    assert_relative_eq!(
                        hessian[i][j],
                        hessian_fd[i][j],
                        epsilon = 1e-3,
                        max_relative = 1e-2
                    );
                }
            }
        }
    }

    #[test]
    fn test_schaffer_n4_optimum() {
        assert_relative_eq!(
            schaffer_n4(&[0_f32, 1.25313_f32]),
            0.29257864,
            epsilon = f32::EPSILON
        );

        let deriv = schaffer_n4_derivative(&[0.0, 1.25313]);
        for i in 0..2 {
            assert_relative_eq!(deriv[i], 0.0, epsilon = 1e-4);
        }
    }

    proptest! {
        #[test]
        fn test_schaffer_n4_derivative_finitediff(a in -100.0..100.0, b in -100.0..100.0) {
            let param = [a, b];
            let derivative = schaffer_n4_derivative(&param);
            let derivative_fd = Vec::from(param).central_diff(&|x| schaffer_n4(&[x[0], x[1]]));
            // println!("1: {derivative:?} at {a}/{b}");
            // println!("2: {derivative_fd:?} at {a}/{b}");
            for i in 0..derivative.len() {
                assert_relative_eq!(
                    derivative[i],
                    derivative_fd[i],
                    epsilon = 1e-3,
                    max_relative = 1e-2,
                );
            }
        }
    }

    proptest! {
        #[test]
        fn test_schaffer_n4_hessian_finitediff(a in -10.0..10.0, b in -10.0..10.0) {
            // Note: prop testing range is smaller than range of function, simply because finitediff
            // has huge errors far away from the optimum.
            let param = [a, b];
            let hessian = schaffer_n4_hessian(&param);
            let hessian_fd =
                Vec::from(param).forward_hessian(&|x| schaffer_n4_derivative(&[x[0], x[1]]).to_vec());
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
                            epsilon = 1e-3,
                            max_relative = 1e-2
                        );
                    }
                }
            }
        }
    }
}
