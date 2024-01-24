// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # McCorminck test function
//!
//! Defined as
//!
//! `f(x_1, x_2) = sin(x_1 + x_2) + (x_1 - x_2)^2 - 1.5*x_1 + 2.5*x_2 + 1`
//!
//! where `x_1 \in [-1.5, 4]` and `x_2 \in [-3, 4]`.
//!
//! The global minimum is at `f(x_1, x_2) = f(-0.54719, -1.54719) = -1.913228`.

use num::{Float, FromPrimitive};

/// McCorminck test function
///
/// Defined as
///
/// `f(x_1, x_2) = (x_1 + x_2).sin() + (x_1 - x_2)^2 - 1.5*x_1 + 2.5*x_2 + 1`
///
/// where `x_1 \in [-1.5, 4]` and `x_2 \in [-3, 4]`.
///
/// The global minimum is at `f(x_1, x_2) = f(-0.54719, -1.54719) = -1.913228`.
pub fn mccorminck<T>(param: &[T; 2]) -> T
where
    T: Float + FromPrimitive,
{
    let [x1, x2] = *param;
    (x1 + x2).sin() + (x1 - x2).powi(2) - T::from_f64(1.5).unwrap() * x1
        + T::from_f64(2.5).unwrap() * x2
        + T::from_f64(1.0).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_mccorminck_optimum() {
        assert_relative_eq!(
            mccorminck(&[-0.54719_f32, -1.54719_f32]),
            -1.9132228,
            epsilon = std::f32::EPSILON
        );
        assert_relative_eq!(
            mccorminck(&[-0.54719_f64, -1.54719_f64]),
            -1.9132229544882274,
            epsilon = std::f32::EPSILON.into()
        );
    }
}
