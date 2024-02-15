// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Eggholder test function
//!
//! Defined as
//!
//! `f(x_1, x_2) = -(x_2 + 47) * sin( sqrt( abs( x_2 + x_1/2 + 47 ) ) ) -
//!                x_1 * sin( sqrt( abs( x_1 - (x_2 + 47) ) ) )`
//!
//! where `x_i \in [-512, 512]`.
//!
//! The global minimum is at * `f(x_1, x_2) = f(512, 404.2319) = -959.6407`.

use num::{Float, FromPrimitive};

/// Eggholder test function
///
/// Defined as
///
/// `f(x_1, x_2) = -(x_2 + 47) * sin( sqrt( abs( x_2 + x_1/2 + 47 ) ) ) -
///                x_1 * sin( sqrt( abs( x_1 - (x_2 + 47) ) ) )`
///
/// where `x_i \in [-512, 512]`.
///
/// The global minimum is at * `f(x_1, x_2) = f(512, 404.2319) = -959.6407`.
pub fn eggholder<T>(param: &[T; 2]) -> T
where
    T: Float + FromPrimitive,
{
    let [x1, x2] = *param;
    let n47 = T::from_f64(47.0).unwrap();
    -(x2 + n47)
        * (x2 + x1 / T::from_f64(2.0).unwrap() + n47)
            .abs()
            .sqrt()
            .sin()
        - x1 * (x1 - (x2 + n47)).abs().sqrt().sin()
}

/// Derivative of Eggholder test function
pub fn eggholder_derivative<T>(param: &[T; 2]) -> [T; 2]
where
    T: Float + FromPrimitive,
{
    let [x1, x2] = *param;

    let eps = T::epsilon();
    let n0 = T::from_f64(0.0).unwrap();
    let n2 = T::from_f64(2.0).unwrap();
    let n4 = T::from_f64(4.0).unwrap();
    let n47 = T::from_f64(47.0).unwrap();

    let x1mx2m47 = x1 - x2 - n47;
    let x1mx2m47abs = x1mx2m47.abs();
    let x1mx2m47abssqrt = x1mx2m47abs.sqrt();
    let x1mx2m47abssqrtsin = x1mx2m47abssqrt.sin();
    let x1mx2m47abssqrtcos = x1mx2m47abssqrt.cos();
    let x1hpx2p47 = x1 / n2 + x2 + n47;
    let x1hpx2p47abs = x1hpx2p47.abs();
    let x1hpx2p47abssqrt = x1hpx2p47abs.sqrt();
    let x1hpx2p47abssqrtsin = x1hpx2p47abssqrt.sin();
    let x1hpx2p47abssqrtcos = x1hpx2p47abssqrt.cos();
    let x2mx1p47 = x2 - x1 + n47;
    let x2mx1p47abs = x2mx1p47.abs();
    let x2mx1p47abssqrt = x2mx1p47abs.sqrt();
    let x2mx1p47abssqrtcos = x2mx1p47abssqrt.cos();

    [
        -x1mx2m47abssqrtsin
            - if x1mx2m47abs <= eps {
                n0
            } else {
                (x1 * x1mx2m47 * x1mx2m47abssqrtcos) / (n2 * x1mx2m47abssqrt.powi(3))
            }
            - if x1hpx2p47abs <= eps {
                n0
            } else {
                ((x2 + n47) * x1hpx2p47abssqrtcos * x1hpx2p47) / (n4 * x1hpx2p47abssqrt.powi(3))
            },
        -x1hpx2p47abssqrtsin
            - if x1hpx2p47abs <= eps {
                n0
            } else {
                ((x2 + n47) * x1hpx2p47 * x1hpx2p47abssqrtcos) / (n2 * x1hpx2p47abssqrt.powi(3))
            }
            - if x2mx1p47abs <= eps {
                n0
            } else {
                (x1 * x2mx1p47 * x2mx1p47abssqrtcos) / (n2 * x2mx1p47abssqrt.powi(3))
            },
    ]
}

/// Hessian of Eggholder test function
///
/// This function can return NaN elements under the following conditions:
///
/// * |x1 - x2 - 47| <= EPS && x1 != 0
/// * |x2 - x1 + 47| <= EPS && x1 != 0
/// * |x1/2 + x2 + 47| <= EPS && |x2 + 47| != 0
pub fn eggholder_hessian<T>(param: &[T; 2]) -> [[T; 2]; 2]
where
    T: Float + FromPrimitive,
{
    let [x1, x2] = *param;

    let eps = T::epsilon();
    let n0 = T::from_f64(0.0).unwrap();
    let n2 = T::from_f64(2.0).unwrap();
    let n3 = T::from_f64(3.0).unwrap();
    let n4 = T::from_f64(4.0).unwrap();
    let n8 = T::from_f64(8.0).unwrap();
    let n16 = T::from_f64(16.0).unwrap();
    let n47 = T::from_f64(47.0).unwrap();

    let x1mx2m47 = x1 - x2 - n47;
    let x1mx2m47abs = x1mx2m47.abs();
    let x1mx2m47abssqrt = x1mx2m47abs.sqrt();
    let x1mx2m47abssqrtsin = x1mx2m47abssqrt.sin();
    let x1mx2m47abssqrtcos = x1mx2m47abssqrt.cos();
    let x1hpx2p47 = x1 / n2 + x2 + n47;
    let x1hpx2p47abs = x1hpx2p47.abs();
    let x1hpx2p47abssqrt = x1hpx2p47abs.sqrt();
    let x1hpx2p47abssqrtsin = x1hpx2p47abssqrt.sin();
    let x1hpx2p47abssqrtcos = x1hpx2p47abssqrt.cos();
    let x2mx1p47 = x2 - x1 + n47;
    let x2mx1p47abs = x2mx1p47.abs();
    let x2mx1p47abssqrt = x2mx1p47abs.sqrt();
    let x2mx1p47abssqrtcos = x2mx1p47abssqrt.cos();
    let x2mx1p47abssqrtsin = x2mx1p47abssqrt.sin();

    let a = if x1mx2m47abs <= eps {
        n0
    } else {
        (x1 * x1mx2m47abssqrtsin) / (n4 * x1mx2m47abs)
            - (x1mx2m47 * x1mx2m47abssqrtcos) / (x1mx2m47abssqrt.powi(3))
            + (n3 * x1 * x1mx2m47.powi(2) * x1mx2m47abssqrtcos) / (n4 * x1mx2m47abssqrt.powi(7))
    } + if x1mx2m47abs <= eps && x1.abs() <= eps {
        n0
    } else {
        -(x1 * x1mx2m47abssqrtcos) / (n2 * x1mx2m47abssqrt.powi(3))
    } + if x1hpx2p47abs <= eps {
        n0
    } else {
        (n3 * (x2 + n47) * x1hpx2p47abssqrtcos * x1hpx2p47.powi(2))
            / (n16 * x1hpx2p47abssqrt.powi(7))
            + ((x2 + n47) * x1hpx2p47abssqrtsin) / (n16 * x1hpx2p47abs)
    } - if x1hpx2p47abs <= eps && (x1 + n47).abs() <= eps {
        n0
    } else {
        ((x2 + n47) * x1hpx2p47abssqrtcos) / (n8 * x1hpx2p47abssqrt.powi(3))
    };

    let b = if x1hpx2p47abs <= eps {
        n0
    } else {
        ((x2 + n47) * x1hpx2p47abssqrtsin) / (n4 * x1hpx2p47abs)
            - (x1hpx2p47 * x1hpx2p47abssqrtcos) / (x1hpx2p47abssqrt.powi(3))
            + (n3 * (x2 + n47) * x1hpx2p47.powi(2) * x1hpx2p47abssqrtcos)
                / (n4 * x1hpx2p47abssqrt.powi(7))
    } - if x1hpx2p47abs <= eps && (x2 + n47).abs() <= eps {
        n0
    } else {
        ((x2 + n47) * x1hpx2p47abssqrtcos) / (n2 * x1hpx2p47abssqrt.powi(3))
    } + if x2mx1p47abs <= eps {
        n0
    } else {
        (x1 * x2mx1p47abssqrtsin) / (n4 * x2mx1p47abs)
            + (n3 * x1 * x2mx1p47.powi(2) * x2mx1p47abssqrtcos) / (n4 * x2mx1p47abssqrt.powi(7))
    } - if x2mx1p47abs <= eps && x1.abs() <= eps {
        n0
    } else {
        (x1 * x2mx1p47abssqrtcos) / (n2 * x2mx1p47abssqrt.powi(3))
    };

    let offdiag = if x1hpx2p47abs <= eps {
        n0
    } else {
        ((x2 + n47) * x1hpx2p47abssqrtsin) / (n8 * x1hpx2p47abs)
            - (x1hpx2p47 * x1hpx2p47abssqrtcos) / (n4 * x1hpx2p47abssqrt.powi(3))
            + (n3 * (x2 + n47) * x1hpx2p47.powi(2) * x1hpx2p47abssqrtcos)
                / (n8 * x1hpx2p47abssqrt.powi(7))
    } - if x1hpx2p47abs <= eps && (x2 + n47).abs() <= eps {
        n0
    } else {
        ((x2 + n47) * x1hpx2p47abssqrtcos) / (n4 * x1hpx2p47abssqrt.powi(3))
    } + if x2mx1p47abs <= eps {
        n0
    } else {
        -(x1 * x2mx1p47 * x2mx1p47abssqrtsin) / (n4 * x2mx1p47 * x2mx1p47abs)
            - (x2mx1p47 * x2mx1p47abssqrtcos) / (n2 * x2mx1p47abssqrt.powi(3))
            - (n3 * x1 * x2mx1p47.powi(2) * x2mx1p47abssqrtcos) / (n4 * x2mx1p47abssqrt.powi(7))
    } + if x2mx1p47abs <= eps && x1.abs() <= eps {
        n0
    } else {
        (x1 * x2mx1p47abssqrtcos) / (n2 * x2mx1p47abssqrt.powi(3))
    };

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
    fn test_eggholder_optimum() {
        assert_relative_eq!(
            eggholder(&[512.0_f32, 404.2319_f32]),
            -959.6407_f32,
            epsilon = f32::EPSILON
        );
        assert_relative_eq!(
            eggholder(&[512.0_f64, 404.2319_f64]),
            -959.6406627106155_f64,
            epsilon = f64::EPSILON
        );
    }

    proptest! {
        #[test]
        fn test_eggholder_derivative_finitediff(a in -512.0..512.0, b in -512.0..512.0) {
            let param = [a, b];
            let derivative = eggholder_derivative(&param);
            let derivative_fd = Vec::from(param).central_diff(&|x| eggholder(&[x[0], x[1]]));
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
        fn test_eggholder_hessian_finitediff(a in -512.0..512.0, b in -512.0..512.0) {
            let param = [a, b];
            let hessian = eggholder_hessian(&param);
            let hessian_fd =
                Vec::from(param).central_hessian(&|x| eggholder_derivative(&[x[0], x[1]]).to_vec());
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
                            epsilon = 1e-4,
                            max_relative = 1e-2
                        );
                    }
                }
            }
        }
    }
}
