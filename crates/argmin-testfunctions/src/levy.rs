// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Levy test function
//!
//! Defined as
//!
//! `f(x_1, x_2, ..., x_n) = sin^2(pi * w1) + \sum_{i=1}^{d-1}(w_i -1)^2 * (1+10*sin^2(pi*wi+1)) +
//! (w_d - 1)^2 * (1 + sin^2(2*pi*w_d))`
//!
//! where `w_i = 1 + (x_i - 1)/4` and `x_i \in [-10, 10]`.
//!
//! The global minimum is at `f(x_1, x_2, ..., x_n) = f(1, 1, ..., 1) = 0`.
//!
//! # Levy test function No. 13
//!
//! Defined as
//!
//! `f(x_1, x_2) = sin^2(3 * pi * x_1) + (x_1 - 1)^2 * (1 + sin^2(3 * pi * x_2)) + (x_2 - 1)^2 *
//! (1 + sin^2(2 * pi * x_2))`
//!
//! where `x_i \in [-10, 10]`.
//!
//! The global minimum is at `f(x_1, x_2) = f(1, 1) = 0`.

use num::{Float, FromPrimitive};
use std::f64::consts::PI;
use std::iter::Sum;

/// Levy test function
///
/// Defined as
///
/// `f(x_1, x_2, ..., x_n) = sin^2(pi * w1) + \sum_{i=1}^{d-1}(w_i -1)^2 * (1+10*sin^2(pi*wi+1)) +
/// (w_d - 1)^2 * (1 + sin^2(2*pi*w_d))`
///
/// where `w_i = 1 + (x_i - 1)/4` and `x_i \in [-10, 10]`.
///
/// The global minimum is at `f(x_1, x_2, ..., x_n) = f(1, 1, ..., 1) = 0`.
pub fn levy<T>(param: &[T]) -> T
where
    T: Float + FromPrimitive + Sum,
{
    let plen = param.len();
    assert!(plen >= 2);

    let n1 = T::from_f64(1.0).unwrap();
    let n2 = T::from_f64(2.0).unwrap();
    let n4 = T::from_f64(4.0).unwrap();
    let n10 = T::from_f64(10.0).unwrap();
    let pi = T::from_f64(PI).unwrap();

    let w = |x: T| n1 + (x - n1) / n4;

    (pi * w(param[0])).sin().powi(2)
        + param[1..(plen - 1)]
            .iter()
            .map(|x| w(*x))
            .map(|wi: T| (wi - n1).powi(2) * (n1 + n10 * (pi * wi + n1).sin().powi(2)))
            .sum()
        + (w(param[plen - 1]) - n1).powi(2) * (n1 + (n2 * pi * w(param[plen - 1])).sin().powi(2))
}

/// Derivative of Levy test function
pub fn levy_derivative<T>(param: &[T]) -> Vec<T>
where
    T: Float + FromPrimitive + Sum,
{
    let d = param.len();
    assert!(d >= 2);

    let n1 = T::from_f64(1.0).unwrap();
    let n2 = T::from_f64(2.0).unwrap();
    let n4 = T::from_f64(4.0).unwrap();
    let n5 = T::from_f64(5.0).unwrap();
    let n8 = T::from_f64(8.0).unwrap();
    let n10 = T::from_f64(10.0).unwrap();
    let n16 = T::from_f64(16.0).unwrap();
    let pi = T::from_f64(PI).unwrap();

    param
        .iter()
        .enumerate()
        .map(|(i, x)| (i, x, pi * ((*x - n1) / n4 + n1)))
        .map(|(i, &x, wp)| {
            if i == 0 {
                pi / n2 * wp.cos() * wp.sin()
            } else if i == d - 1 {
                ((n2 * wp).sin().powi(2) + n1) * (x - n1) / n8
                    + pi / n16 * (n2 * wp).cos() * (n2 * wp).sin() * (x - n1).powi(2)
            } else {
                (n10 * (wp + n1).sin().powi(2) + n1) * (x - n1) / n8
                    + n5 / n16 * pi * (wp + n1).cos() * (wp + n1).sin() * (x - n1).powi(2)
            }
        })
        .collect()
}

/// Derivative of Levy test function
///
/// This is the const generics version, which requires the number of parameters to be known
/// at compile time.
pub fn levy_derivative_const<const N: usize, T>(param: &[T; N]) -> [T; N]
where
    T: Float + FromPrimitive + Sum,
{
    assert!(N >= 2);

    let n1 = T::from_f64(1.0).unwrap();
    let n0 = T::from_f64(0.0).unwrap();
    let n2 = T::from_f64(2.0).unwrap();
    let n4 = T::from_f64(4.0).unwrap();
    let n5 = T::from_f64(5.0).unwrap();
    let n8 = T::from_f64(8.0).unwrap();
    let n10 = T::from_f64(10.0).unwrap();
    let n16 = T::from_f64(16.0).unwrap();
    let pi = T::from_f64(PI).unwrap();

    let mut out = [n0; N];

    param
        .iter()
        .zip(out.iter_mut())
        .enumerate()
        .map(|(i, (x, o))| (i, x, pi * ((*x - n1) / n4 + n1), o))
        .map(|(i, &x, wp, o)| {
            *o = if i == 0 {
                pi / n2 * wp.cos() * wp.sin()
            } else if i == N - 1 {
                ((n2 * wp).sin().powi(2) + n1) * (x - n1) / n8
                    + pi / n16 * (n2 * wp).cos() * (n2 * wp).sin() * (x - n1).powi(2)
            } else {
                (n10 * (wp + n1).sin().powi(2) + n1) * (x - n1) / n8
                    + n5 / n16 * pi * (wp + n1).cos() * (wp + n1).sin() * (x - n1).powi(2)
            }
        })
        .count();
    out
}

/// Hessian of Levy test function
pub fn levy_hessian<T>(param: &[T]) -> Vec<Vec<T>>
where
    T: Float + FromPrimitive + Sum,
{
    let d = param.len();
    assert!(d >= 2);

    let x = param;

    let n0 = T::from_f64(0.0).unwrap();
    let n1 = T::from_f64(1.0).unwrap();
    let n2 = T::from_f64(2.0).unwrap();
    let n4 = T::from_f64(4.0).unwrap();
    let n5 = T::from_f64(5.0).unwrap();
    let n6 = T::from_f64(6.0).unwrap();
    let n8 = T::from_f64(8.0).unwrap();
    let n10 = T::from_f64(10.0).unwrap();
    let n32 = T::from_f64(32.0).unwrap();
    let n64 = T::from_f64(64.0).unwrap();
    let pi = T::from_f64(PI).unwrap();
    let pi2 = T::from_f64(PI.powi(2)).unwrap();

    let mut out = vec![vec![n0; d]; d];

    for i in 0..d {
        let xin1 = x[i] - n1;
        let wp = pi * (xin1 / n4 + n1);
        out[i][i] = if i == 0 {
            pi2 / n8 * (wp.cos().powi(2) - wp.sin().powi(2))
        } else if i == d - 1 {
            -(n4 * pi * xin1 * (pi * x[i]).sin() + (pi2 * xin1.powi(2) - n2) * (pi * x[i]).cos()
                - n6)
                / n32
        } else {
            let wp1cos = (wp + n1).cos();
            let wp1sin = (wp + n1).sin();
            n5 / n4 * pi * wp1cos * wp1sin * xin1
                + n5 / n64 * pi2 * xin1.powi(2) * (wp1cos.powi(2) - wp1sin.powi(2))
                + (n10 * wp1sin.powi(2) + n1) / n8
        }
    }

    out
}

/// Hessian of Levy test function
///
/// This is the const generics version, which requires the number of parameters to be known
/// at compile time.
pub fn levy_hessian_const<const N: usize, T>(param: &[T; N]) -> [[T; N]; N]
where
    T: Float + FromPrimitive + Sum,
{
    assert!(N >= 2);

    let x = param;

    let n0 = T::from_f64(0.0).unwrap();
    let n1 = T::from_f64(1.0).unwrap();
    let n2 = T::from_f64(2.0).unwrap();
    let n4 = T::from_f64(4.0).unwrap();
    let n5 = T::from_f64(5.0).unwrap();
    let n6 = T::from_f64(6.0).unwrap();
    let n8 = T::from_f64(8.0).unwrap();
    let n10 = T::from_f64(10.0).unwrap();
    let n32 = T::from_f64(32.0).unwrap();
    let n64 = T::from_f64(64.0).unwrap();
    let pi = T::from_f64(PI).unwrap();
    let pi2 = T::from_f64(PI.powi(2)).unwrap();

    let mut out = [[n0; N]; N];

    for i in 0..N {
        let xin1 = x[i] - n1;
        let wp = pi * (xin1 / n4 + n1);
        out[i][i] = if i == 0 {
            pi2 / n8 * (wp.cos().powi(2) - wp.sin().powi(2))
        } else if i == N - 1 {
            -(n4 * pi * xin1 * (pi * x[i]).sin() + (pi2 * xin1.powi(2) - n2) * (pi * x[i]).cos()
                - n6)
                / n32
        } else {
            let wp1cos = (wp + n1).cos();
            let wp1sin = (wp + n1).sin();
            n5 / n4 * pi * wp1cos * wp1sin * xin1
                + n5 / n64 * pi2 * xin1.powi(2) * (wp1cos.powi(2) - wp1sin.powi(2))
                + (n10 * wp1sin.powi(2) + n1) / n8
        }
    }

    out
}

/// Levy test function No. 13
///
/// Defined as
///
/// `f(x_1, x_2) = sin^2(3 * pi * x_1) + (x_1 - 1)^2 * (1 + sin^2(3 * pi * x_2)) + (x_2 - 1)^2 *
/// (1 + sin^2(2 * pi * x_2))`
///
/// where `x_i \in [-10, 10]`.
///
/// The global minimum is at `f(x_1, x_2) = f(1, 1) = 0`.
pub fn levy_n13<T>(param: &[T; 2]) -> T
where
    T: Float + FromPrimitive + Sum,
{
    let [x1, x2] = *param;

    let n1 = T::from_f64(1.0).unwrap();
    let n2 = T::from_f64(2.0).unwrap();
    let n3 = T::from_f64(3.0).unwrap();
    let pi = T::from_f64(PI).unwrap();

    (n3 * pi * x1).sin().powi(2)
        + (x1 - n1).powi(2) * (n1 + (n3 * pi * x2).sin().powi(2))
        + (x2 - n1).powi(2) * (n1 + (n2 * pi * x2).sin().powi(2))
}

/// Derivative of Levy test function No. 13
pub fn levy_n13_derivative<T>(param: &[T; 2]) -> [T; 2]
where
    T: Float + FromPrimitive + Sum,
{
    let [x1, x2] = *param;

    let n1 = T::from_f64(1.0).unwrap();
    let n2 = T::from_f64(2.0).unwrap();
    let n3 = T::from_f64(3.0).unwrap();
    let n4 = T::from_f64(4.0).unwrap();
    let n6 = T::from_f64(6.0).unwrap();
    let pi = T::from_f64(PI).unwrap();

    let x1t3 = n3 * pi * x1;
    let x2t3 = n3 * pi * x2;
    let x2t2 = n2 * pi * x2;
    let x1t3s = x1t3.sin();
    let x1t3c = x1t3.cos();
    let x2t3s = x2t3.sin();
    let x2t3c = x2t3.cos();
    let x2t3s2 = x2t3s.powi(2);
    let x2t2s = x2t2.sin();
    let x2t2c = x2t2.cos();
    let x2t2s2 = x2t2s.powi(2);

    [
        n6 * pi * x1t3c * x1t3s + n2 * (x2t3s2 + n1) * (x1 - n1),
        n6 * pi * (x1 - n1).powi(2) * x2t3c * x2t3s
            + n2 * (x2 - n1) * (x2t2s2 + n1)
            + n4 * pi * (x2 - n1).powi(2) * x2t2c * x2t2s,
    ]
}

/// Hessian of Levy test function No. 13
pub fn levy_n13_hessian<T>(param: &[T; 2]) -> [[T; 2]; 2]
where
    T: Float + FromPrimitive + Sum,
{
    let [x1, x2] = *param;

    let n1 = T::from_f64(1.0).unwrap();
    let n2 = T::from_f64(2.0).unwrap();
    let n3 = T::from_f64(3.0).unwrap();
    let n8 = T::from_f64(8.0).unwrap();
    let n12 = T::from_f64(12.0).unwrap();
    let n16 = T::from_f64(16.0).unwrap();
    let n18 = T::from_f64(18.0).unwrap();
    let pi = T::from_f64(PI).unwrap();
    let pi2 = T::from_f64(PI.powi(2)).unwrap();

    let x1t3 = n3 * pi * x1;
    let x2t3 = n3 * pi * x2;
    let x2t2 = n2 * pi * x2;
    let x1t3s = x1t3.sin();
    let x1t3c = x1t3.cos();
    let x2t3s = x2t3.sin();
    let x2t3c = x2t3.cos();
    let x1t3s2 = x1t3s.powi(2);
    let x1t3c2 = x1t3c.powi(2);
    let x2t3s2 = x2t3s.powi(2);
    let x2t3c2 = x2t3c.powi(2);
    let x2t2s = x2t2.sin();
    let x2t2c = x2t2.cos();
    let x2t2s2 = x2t2s.powi(2);
    let x2t2c2 = x2t2c.powi(2);

    let a = n18 * pi2 * (-x1t3s2 + x1t3c2) + n2 * (x2t3s2 + n1);
    let b = n18 * pi2 * (x1 - n1).powi(2) * (-x2t3s2 + x2t3c2)
        + n2 * (x2t2s2 + n1)
        + n8 * pi2 * (x2 - n1).powi(2) * (-x2t2s2 + x2t2c2)
        + n16 * pi * (x2 - n1) * x2t2s * x2t2c;
    let offdiag = n12 * pi * (x1 - n1) * x2t3c * x2t3s;

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
    fn test_levy_optimum() {
        assert_relative_eq!(levy(&[1_f32, 1_f32, 1_f32]), 0.0, epsilon = f32::EPSILON);
        assert_relative_eq!(levy(&[1_f64, 1_f64, 1_f64]), 0.0, epsilon = f64::EPSILON);

        let deriv = levy_derivative(&[1_f64, 1_f64, 1_f64]);
        for i in 0..2 {
            assert_relative_eq!(deriv[i], 0.0, epsilon = 1e-12, max_relative = 1e-12);
        }

        let deriv = levy_derivative_const(&[1_f64, 1_f64, 1_f64]);
        for i in 0..2 {
            assert_relative_eq!(deriv[i], 0.0, epsilon = 1e-12, max_relative = 1e-12);
        }
    }

    #[test]
    fn test_levy_n13_optimum() {
        assert_relative_eq!(levy_n13(&[1_f32, 1_f32]), 0.0, epsilon = f32::EPSILON);
        assert_relative_eq!(levy_n13(&[1_f64, 1_f64]), 0.0, epsilon = f64::EPSILON);

        let deriv = levy_n13_derivative(&[1_f64, 1_f64]);
        for i in 0..2 {
            assert_relative_eq!(deriv[i], 0.0, epsilon = 1e-12, max_relative = 1e-12);
        }
    }

    #[test]
    #[should_panic]
    fn test_levy_param_length() {
        levy(&[0.0_f32]);
    }

    proptest! {
        #[test]
        fn test_levy_n13_derivative_finitediff(a in -10.0..10.0, b in -10.0..10.0) {
            let param = [a, b];
            let derivative = levy_n13_derivative(&param);
            let derivative_fd = Vec::from(param).central_diff(&|x| levy_n13(&[x[0], x[1]]));
            // println!("1: {derivative:?} at {a}/{b}");
            // println!("2: {derivative_fd:?} at {a}/{b}");
            for i in 0..derivative.len() {
                assert_relative_eq!(
                    derivative[i],
                    derivative_fd[i],
                    epsilon = 1e-5,
                    max_relative = 1e-2,
                );
            }
        }
    }

    proptest! {
        #[test]
        fn test_levy_n13_hessian_finitediff(a in -10.0..10.0, b in -10.0..10.0) {
            let param = [a, b];
            let hessian = levy_n13_hessian(&param);
            let hessian_fd =
                Vec::from(param).central_hessian(&|x| levy_n13_derivative(&[x[0], x[1]]).to_vec());
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

    proptest! {
        #[test]
        fn test_levy_derivative_finitediff(a in -10.0..10.0, b in -10.0..10.0, c in -10.0..10.0) {
            let param = [a, b, c];
            let derivative = levy_derivative(&param);
            let derivative_fd = Vec::from(param).central_diff(&|x| levy(&[x[0], x[1], x[2]]));
            // println!("1: {derivative:?} at {param:?}");
            // println!("2: {derivative_fd:?} at {param:?}");
            for i in 0..derivative.len() {
                assert_relative_eq!(
                    derivative[i],
                    derivative_fd[i],
                    epsilon = 1e-5,
                    max_relative = 1e-2,
                );
            }
        }
    }

    proptest! {
        #[test]
        fn test_levy_derivative_const_finitediff(a in -10.0..10.0, b in -10.0..10.0, c in -10.0..10.0) {
            let param = [a, b, c];
            let derivative = levy_derivative_const(&param);
            let derivative_fd = Vec::from(param).central_diff(&|x| levy(&[x[0], x[1], x[2]]));
            // println!("1: {derivative:?} at {param:?}");
            // println!("2: {derivative_fd:?} at {param:?}");
            for i in 0..derivative.len() {
                assert_relative_eq!(
                    derivative[i],
                    derivative_fd[i],
                    epsilon = 1e-5,
                    max_relative = 1e-2,
                );
            }
        }
    }

    proptest! {
        #[test]
        fn test_levy_hessian_finitediff(a in -10.0..10.0, b in -10.0..10.0, c in -10.0..10.0) {
            let param = [a, b, c];
            let hessian = levy_hessian(&param);
            let hessian_fd = Vec::from(param).central_hessian(&|x| levy_derivative(&x).to_vec());
            let n = hessian.len();
            // println!("1: {hessian:?} at {param:?}");
            // println!("2: {hessian_fd:?} at {param:?}");
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

    proptest! {
        #[test]
        fn test_levy_hessian_const_finitediff(a in -10.0..10.0, b in -10.0..10.0, c in -10.0..10.0) {
            let param = [a, b, c];
            let hessian = levy_hessian_const(&param);
            let hessian_fd = Vec::from(param).central_hessian(&|x| levy_derivative(&x).to_vec());
            let n = hessian.len();
            // println!("1: {hessian:?} at {param:?}");
            // println!("2: {hessian_fd:?} at {param:?}");
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
