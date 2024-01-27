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

    let w = |x: T| n1 - (x - n1) / n4;

    (pi * w(param[0])).sin().powi(2)
        + param[1..(plen - 1)]
            .iter()
            .map(|x| w(*x))
            .map(|wi: T| (wi - n1).powi(2) * (n1 + n10 * (pi * wi + n1)))
            .sum()
        + (w(param[plen - 1]) - n1).powi(2) * (n1 + (n2 * pi * w(param[plen - 1])).sin().powi(2))
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::{f32, f64};

    #[test]
    fn test_levy_optimum() {
        assert_relative_eq!(levy(&[1_f32, 1_f32, 1_f32]), 0.0, epsilon = f32::EPSILON);
        assert_relative_eq!(levy(&[1_f64, 1_f64, 1_f64]), 0.0, epsilon = f64::EPSILON);
    }

    #[test]
    fn test_levy_n13_optimum() {
        assert_relative_eq!(levy_n13(&[1_f32, 1_f32]), 0.0, epsilon = f32::EPSILON);
        assert_relative_eq!(levy_n13(&[1_f64, 1_f64]), 0.0, epsilon = f64::EPSILON);
    }

    #[test]
    #[should_panic]
    fn test_levy_param_length() {
        levy(&[0.0_f32]);
    }
}
