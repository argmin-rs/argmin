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
///                (2.625 - x_1 + x1 * x_2^3)^2`
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::{f32, f64};

    #[test]
    fn test_beale_optimum() {
        assert_relative_eq!(beale(&[3.0_f32, 0.5_f32]), 0.0, epsilon = f32::EPSILON);
        assert_relative_eq!(beale(&[3.0_f64, 0.5_f64]), 0.0, epsilon = f64::EPSILON);
    }
}
