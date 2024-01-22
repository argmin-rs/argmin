// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Rosenbrock function
//!
//! In 2D, it is defined as
//!
//! `f(x_1, x_2) = (a - x_1)^2 + b * (x_2 - x_1^2)^2`
//!
//! where `x_i \in (-\infty, \infty)`. The parameters a and b usually are: `a = 1` and `b = 100`.
//!
//! The multidimensional Rosenbrock function is defined as:
//!
//! `f(x_1, x_2, ..., x_n) = \sum_{i=1}^{n-1} \left[ (a - x_i)^2 + b * (x_{i+1} - x_i^2)^2 \right]`
//!
//! The minimum is at `f(x_1, x_2, ..., x_n) = f(1, 1, ..., 1) = 0`.

use num::{Float, FromPrimitive};
use std::fmt::Debug;
use std::iter::Sum;

/// Multidimensional Rosenbrock test function
///
/// Defined as
///
/// `f(x_1, x_2, ..., x_n) = \sum_{i=1}^{n-1} \left[ (a - x_i)^2 + b * (x_{i+1} - x_i^2)^2 \right]`
///
/// where `x_i \in (-\infty, \infty)`. The parameters a and b usually are: `a = 1` and `b = 100`.
///
/// The global minimum is at `f(x_1, x_2, ..., x_n) = f(1, 1, ..., 1) = 0`.
pub fn rosenbrock<T: Float + FromPrimitive + Sum + Debug>(param: &[T], a: T, b: T) -> T {
    param
        .iter()
        .zip(param.iter().skip(1))
        .map(|(&xi, &xi1)| (a - xi).powi(2) + b * (xi1 - xi.powi(2)).powi(2))
        .sum()
}

/// 2D Rosenbrock test function
///
/// Defined as
///
/// `f(x_1, x_2) = (a - x_1)^2 + b * (x_2 - x_1^2)^2`
///
/// where `x_i \in (-\infty, \infty)`. The parameters a and b usually are: `a = 1` and `b = 100`.
///
/// For 2D problems, this function is much faster than `rosenbrock`.
///
/// The global minimum is at `f(x_1, x_2) = f(1, 1) = 0`.
pub fn rosenbrock_2d<T: Float + FromPrimitive>(param: &[T], a: T, b: T) -> T {
    if let [x, y] = *param {
        (a - x).powi(2) + b * (y - x.powi(2)).powi(2)
    } else {
        panic!("rosenbrock_2d only works for a parameter vector with two values.");
    }
}

/// Derivative of 2D Rosenbrock function
pub fn rosenbrock_2d_derivative<T: Float + FromPrimitive>(param: &[T], a: T, b: T) -> Vec<T> {
    let num2 = T::from_f64(2.0).unwrap();
    let num4 = T::from_f64(4.0).unwrap();
    if let [x, y] = *param {
        let mut out = Vec::with_capacity(2);
        out.push(-num2 * a + num4 * b * x.powi(3) - num4 * b * x * y + num2 * x);
        out.push(num2 * b * (y - x.powi(2)));
        out
    } else {
        panic!("rosenbrock function only accepts 2 parameters.");
    }
}

/// Hessian of 2D Rosenbrock function
pub fn rosenbrock_2d_hessian<T: Float + FromPrimitive>(param: &[T], _a: T, b: T) -> Vec<T> {
    let num2 = T::from_f64(2.0).unwrap();
    let num4 = T::from_f64(4.0).unwrap();
    let num12 = T::from_f64(12.0).unwrap();
    if let [x, y] = *param {
        let mut out = Vec::with_capacity(4);
        // d/dxdx
        out.push(num12 * b * x.powi(2) - num4 * b * y + num2);
        // d/dxdy
        out.push(-num4 * b * x);
        // d/dydx
        out.push(-num4 * b * x);
        // d/dydy
        out.push(num2 * b);
        out
    } else {
        panic!("rosenbrock_hessian only accepts 2 parameters.");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std;

    #[test]
    fn test_rosenbrock_optimum_2d() {
        assert!(rosenbrock(&[1.0_f32, 1.0_f32], 1.0, 100.0).abs() < std::f32::EPSILON);
        assert!(rosenbrock(&[1.0, 1.0], 1.0, 100.0).abs() < std::f64::EPSILON);
    }

    #[test]
    fn test_rosenbrock_derivative() {
        let res: Vec<f64> = rosenbrock_2d_derivative(&[1.0, 1.0], 1.0, 100.0);
        for elem in &res {
            assert!((elem - 0.0).abs() < std::f64::EPSILON);
        }
        let res: Vec<f32> = rosenbrock_2d_derivative(&[1.0_f32, 1.0_f32], 1.0_f32, 100.0_f32);
        for elem in &res {
            assert!((elem - 0.0).abs() < std::f32::EPSILON);
        }
    }

    #[test]
    fn test_rosenbrock_optimum_3d() {
        assert!(rosenbrock(&[1.0, 1.0, 1.0], 1.0, 100.0).abs() < std::f64::EPSILON);
    }

    #[test]
    #[should_panic]
    fn test_rosenbrock_2d_with_nd() {
        rosenbrock_2d(&[1.0, 1.0, 1.0], 1.0, 100.0);
    }
}
