// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Booth test function
//!
//! Defined as
//!
//! `f(x_1, x_2) = (x_1 + 2*x_2 - 7)^2 + (2*x_1 + x_2 - 5)^2
//!
//! where `x_i \in [-10, 10]`.
//!
//! The global minimum is at `f(x_1, x_2) = f(1, 3) = 0`.

use num::{Float, FromPrimitive};

/// Booth test function
///
/// Defined as
///
/// `f(x_1, x_2) = (x_1 + 2*x_2 - 7)^2 + (2*x_1 + x_2 - 5)^2
///
/// where `x_i \in [-10, 10]`.
///
/// The global minimum is at `f(x_1, x_2) = f(1, 3) = 0`.
pub fn booth<T>(param: &[T; 2]) -> T
where
    T: Float + FromPrimitive,
{
    let [x1, x2] = *param;
    (x1 + T::from_f64(2.0).unwrap() * x2 - T::from_f64(7.0).unwrap()).powi(2)
        + (T::from_f64(2.0).unwrap() * x1 + x2 - T::from_f64(5.0).unwrap()).powi(2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::{f32, f64};

    #[test]
    fn test_booth_optimum() {
        assert_relative_eq!(booth(&[1_f32, 3_f32]), 0.0, epsilon = f32::EPSILON);
        assert_relative_eq!(booth(&[1_f64, 3_f64]), 0.0, epsilon = f64::EPSILON);
    }
}
