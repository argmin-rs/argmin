// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Three-hump camel test function
//!
//! Defined as
//!
//! `f(x_1, x_2) = 2*x_1^2 - 1.05*x_1^4 + x_1^6/6 + x_1*x_2 + x_2^2`
//!
//! where `x_i \in [-5, 5]`.
//!
//! The global minimum is at `f(x_1, x_2) = f(0, 0) = 0`.

use num::{Float, FromPrimitive};

/// Three-hump camel test function
///
/// Defined as
///
/// `f(x_1, x_2) = 2*x_1^2 - 1.05*x_1^4 + x_1^6/6 + x_1*x_2 + x_2^2`
///
/// where `x_i \in [-5, 5]`.
///
/// The global minimum is at `f(x_1, x_2) = f(0, 0) = 0`.
pub fn threehumpcamel<T>(param: &[T; 2]) -> T
where
    T: Float + FromPrimitive,
{
    let [x1, x2] = *param;

    T::from_f64(2.0).unwrap() * x1.powi(2) - T::from_f64(1.05).unwrap() * x1.powi(4)
        + x1.powi(6) / T::from_f64(6.0).unwrap()
        + x1 * x2
        + x2.powi(2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::{f32, f64};

    #[test]
    fn test_threehumpcamel_optimum() {
        assert_relative_eq!(
            threehumpcamel(&[0.0_f32, 0.0_f32]),
            0.0,
            epsilon = f32::EPSILON
        );
        assert_relative_eq!(
            threehumpcamel(&[0.0_f64, 0.0_f64]),
            0.0,
            epsilon = f64::EPSILON
        );
    }
}
