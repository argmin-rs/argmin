// Copyright 2018-2020 argmin developers
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
pub fn booth<T: Float + FromPrimitive>(param: &[T]) -> T {
    assert!(param.len() == 2);
    let (x1, x2) = (param[0], param[1]);
    (x1 + T::from_f64(2.0).unwrap() * x2 - T::from_f64(7.0).unwrap()).powi(2)
        + (T::from_f64(2.0).unwrap() * x1 + x2 - T::from_f64(5.0).unwrap()).powi(2)
}

mod tests {
    #[test]
    fn test_booth_optimum() {
        assert!((::booth(&[1_f32, 3_f32])).abs() < ::std::f32::EPSILON);
        assert!((::booth(&[1_f64, 3_f64])).abs() < ::std::f64::EPSILON);
    }

    #[test]
    #[should_panic]
    fn test_booth_param_length() {
        ::booth(&[0.0_f32, -1.0_f32, 0.1_f32]);
    }
}
