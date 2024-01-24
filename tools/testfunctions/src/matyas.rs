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
pub fn matyas<T: Float + FromPrimitive>(param: &[T]) -> T {
    assert!(param.len() == 2);
    let (x1, x2) = (param[0], param[1]);
    T::from_f64(0.26).unwrap() * (x1.powi(2) + x2.powi(2)) - T::from_f64(0.48).unwrap() * x1 * x2
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{f32, f64};

    #[test]
    fn test_matyas_optimum() {
        assert!((matyas(&[0_f32, 0_f32])).abs() < f32::EPSILON);
        assert!((matyas(&[0_f64, 0_f64])).abs() < f64::EPSILON);
    }

    #[test]
    #[should_panic]
    fn test_matyas_param_length() {
        matyas(&[0.0_f32, -1.0_f32, 0.1_f32]);
    }
}
