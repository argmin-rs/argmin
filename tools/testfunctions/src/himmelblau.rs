// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Himmelblau test function
//!
//! Defined as
//!
//! `f(x_1, x_2) = (x_1^2 + x_2 - 11)^2 + (x_1 + x_2^2 - 7)^2`
//!
//! where `x_i \in [-5, 5]`.
//!
//! The global minima are at
//!  * `f(x_1, x_2) = f(3, 2) = 0`.
//!  * `f(x_1, x_2) = f(-2.805118, 3.131312) = 0`.
//!  * `f(x_1, x_2) = f(-3.779310, -3.283186) = 0`.
//!  * `f(x_1, x_2) = f(3.584428, -1.848126) = 0`.

use num::{Float, FromPrimitive};

/// Himmelblau test function
///
/// Defined as
///
/// `f(x_1, x_2) = (x_1^2 + x_2 - 11)^2 + (x_1 + x_2^2 - 7)^2`
///
/// where `x_i \in [-5, 5]`.
///
/// The global minima are at
///  * `f(x_1, x_2) = f(3, 2) = 0`.
///  * `f(x_1, x_2) = f(-2.805118, 3.131312) = 0`.
///  * `f(x_1, x_2) = f(-3.779310, -3.283186) = 0`.
///  * `f(x_1, x_2) = f(3.584428, -1.848126) = 0`.
pub fn himmelblau<T: Float + FromPrimitive>(param: &[T]) -> T {
    assert!(param.len() == 2);
    let (x1, x2) = (param[0], param[1]);
    (x1.powi(2) + x2 - T::from_f64(11.0).unwrap()).powi(2)
        + (x1 + x2.powi(2) - T::from_f64(7.0).unwrap()).powi(2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{f32, f64};

    #[test]
    fn test_himmelblau_optimum() {
        assert!((himmelblau(&[3.0_f32, 2.0_f32])).abs() < f32::EPSILON);
        assert!((himmelblau(&[-2.805118_f32, 3.131312_f32])).abs() < f32::EPSILON);
        assert!((himmelblau(&[-3.779310_f32, -3.283186_f32])).abs() < f32::EPSILON);
        assert!((himmelblau(&[3.584428_f32, -1.848126_f32])).abs() < f32::EPSILON);

        // Since I don't know the 64bit location of the minima,the f64 version cannot be reliably
        // tested without allowing an error several magnitudes larger than EPSILON.
        assert!((himmelblau(&[3.0_f64, 2.0_f64])).abs() < f64::EPSILON);
        // assert!((::himmelblau(&[-2.805118_f64, 3.131312_f64])).abs() < ::std::f64::EPSILON);
        // assert!((::himmelblau(&[-3.779310_f64, -3.283186_f64])).abs() < ::std::f64::EPSILON);
        // assert!((::himmelblau(&[3.584428_f64, -1.848126_f64])).abs() < ::std::f64::EPSILON);
    }

    #[test]
    #[should_panic]
    fn test_himmelblau_param_length() {
        himmelblau(&[0.0_f32, -1.0_f32, 0.1_f32]);
    }
}
