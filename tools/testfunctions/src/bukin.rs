// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Bukin test function No. 6
//!
//! Defined as
//!
//! `f(x_1, x_2) = 100*\sqrt{|x_2 - 0.01*x_1^2|} + 0.01 * |x_1 + 10|`
//!
//! where `x_1 \in [-15, -5]` and `x_2 \in [-3, 3]`.
//!
//! The global minimum is at `f(x_1, x_2) = f(-10, 1) = 0`.

use num::{Float, FromPrimitive};

/// Bukin test function No. 6
///
/// Defined as
///
/// `f(x_1, x_2) = 100*\sqrt{|x_2 - 0.01*x_1^2|} + 0.01 * |x_1 + 10|`
///
/// where `x_1 \in [-15, -5]` and `x_2 \in [-3, 3]`.
///
/// The global minimum is at `f(x_1, x_2) = f(-10, 1) = 0`.
pub fn bukin_n6<T: Float + FromPrimitive>(param: &[T]) -> T {
    assert!(param.len() == 2);
    let (x1, x2) = (param[0], param[1]);
    T::from_f64(100.0).unwrap() * (x2 - T::from_f64(0.01).unwrap() * x1.powi(2)).abs().sqrt()
        + T::from_f64(0.01).unwrap() * (x1 + T::from_f64(10.0).unwrap()).abs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{f32, f64};

    #[test]
    fn test_bukin_n6_optimum() {
        assert!((bukin_n6(&[-10_f32, 1_f32])).abs() < f32::EPSILON);
        assert!((bukin_n6(&[-10_f64, 1_f64])).abs() < f64::EPSILON);
    }

    #[test]
    #[should_panic]
    fn test_bukin_n6_param_length() {
        bukin_n6(&[0.0_f32, -1.0_f32, 0.1_f32]);
    }
}
