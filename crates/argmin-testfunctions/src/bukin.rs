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
pub fn bukin_n6<T>(param: &[T; 2]) -> T
where
    T: Float + FromPrimitive,
{
    let [x1, x2] = *param;
    let n001 = T::from_f64(0.01).unwrap();
    let n10 = T::from_f64(10.0).unwrap();
    let n100 = T::from_f64(100.0).unwrap();
    n100 * (x2 - n001 * x1.powi(2)).abs().sqrt() + n001 * (x1 + n10).abs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::{f32, f64};

    #[test]
    fn test_bukin_n6_optimum() {
        assert_relative_eq!(bukin_n6(&[-10_f32, 1_f32]), 0.0, epsilon = f32::EPSILON);
        assert_relative_eq!(bukin_n6(&[-10_f64, 1_f64]), 0.0, epsilon = f64::EPSILON);
    }
}
