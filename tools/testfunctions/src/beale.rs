// Copyright 2018-2020 argmin developers
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
pub fn beale<T: Float + FromPrimitive>(param: &[T]) -> T {
    assert!(param.len() == 2);
    let (x1, x2) = (param[0], param[1]);
    (T::from_f64(1.5).unwrap() - x1 + x1 * x2).powi(2)
        + (T::from_f64(2.25).unwrap() - x1 + x1 * (x2.powi(2))).powi(2)
        + (T::from_f64(2.625).unwrap() - x1 + x1 * (x2.powi(3))).powi(2)
}

mod tests {
    #[test]
    fn test_beale_optimum() {
        assert!(::beale(&[3.0_f32, 0.5_f32]).abs() < ::std::f32::EPSILON);
        assert!(::beale(&[3.0_f64, 0.5_f64]).abs() < ::std::f64::EPSILON);
    }

    #[test]
    #[should_panic]
    fn test_beale_param_length() {
        ::beale(&[0.0_f32, -1.0_f32, 0.1_f32]);
    }
}
