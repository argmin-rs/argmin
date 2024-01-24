// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Matyas test function
//!
//! Defined as
//!
//! `f(x_1, x_2) = 0.26 * (x_1^2 + x_2^2) - 0.48 * x_1 * x_2`
//!
//! where `x_i \in [-10, 10]`.
//!
//! The global minimum is at `f(x_1, x_2) = f(0, 0) = 0`.

use num::{Float, FromPrimitive};

/// Matyas test function
///
/// Defined as
///
/// `f(x_1, x_2) = 0.26 * (x_1^2 + x_2^2) - 0.48 * x_1 * x_2`
///
/// where `x_i \in [-10, 10]`.
///
/// The global minimum is at `f(x_1, x_2) = f(0, 0) = 0`.
pub fn matyas<T>(param: &[T; 2]) -> T
where
    T: Float + FromPrimitive,
{
    let [x1, x2] = *param;

    let n026 = T::from_f64(0.26).unwrap();
    let n048 = T::from_f64(0.48).unwrap();

    n026 * (x1.powi(2) + x2.powi(2)) - n048 * x1 * x2
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::{f32, f64};

    #[test]
    fn test_matyas_optimum() {
        assert_relative_eq!(matyas(&[0_f32, 0_f32]), 0.0, epsilon = f32::EPSILON);
        assert_relative_eq!(matyas(&[0_f64, 0_f64]), 0.0, epsilon = f64::EPSILON);
    }
}
