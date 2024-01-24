// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Goldstein-Price test function
//!
//! Defined as
//!
//! `f(x_1, x_2) = [1 + (x_1 + x_2 + 1)^2 * (19 - 14*x_2 + + 3*x_1^2 - 14*x_2 6*x_1*x_2 + 3*x_2^2)]
//!                * [30 + (2*x_1 - 3*x_2)^2(18 - 32 * x_1 + 12* x_1^2 + 48 * x_2 -
//!                   36 * x_1 * x_2 + 27 * x_2^2) ]`
//!
//! where `x_i \in [-2, 2]`.
//!
//! The global minimum is at `f(x_1, x_2) = f(0, -1) = 3`.

use num::{Float, FromPrimitive};

/// Goldstein-Price test function
///
/// Defined as
///
/// `f(x_1, x_2) = [1 + (x_1 + x_2 + 1)^2 * (19 - 14*x_2 + 3*x_1^2 - 14*x_2 6*x_1*x_2 + 3*x_2^2)]
///                * [30 + (2*x_1 - 3*x_2)^2(18 - 32 * x_1 + 12* x_1^2 + 48 * x_2 -
///                   36 * x_1 * x_2 + 27 * x_2^2) ]`
///
/// where `x_i \in [-2, 2]`.
///
/// The global minimum is at `f(x_1, x_2) = f(0, -1) = 3`.
pub fn goldsteinprice<T: Float + FromPrimitive>(param: &[T]) -> T {
    assert!(param.len() == 2);
    let (x1, x2) = (param[0], param[1]);
    (T::from_f64(1.0).unwrap()
        + (x1 + x2 + T::from_f64(1.0).unwrap()).powi(2)
            * (T::from_f64(19.0).unwrap() - T::from_f64(14.0).unwrap() * (x1 + x2)
                + T::from_f64(3.0).unwrap() * (x1.powi(2) + x2.powi(2))
                + T::from_f64(6.0).unwrap() * x1 * x2))
        * (T::from_f64(30.0).unwrap()
            + (T::from_f64(2.0).unwrap() * x1 - T::from_f64(3.0).unwrap() * x2).powi(2)
                * (T::from_f64(18.0).unwrap() - T::from_f64(32.0).unwrap() * x1
                    + T::from_f64(12.0).unwrap() * x1.powi(2)
                    + T::from_f64(48.0).unwrap() * x2
                    - T::from_f64(36.0).unwrap() * x1 * x2
                    + T::from_f64(27.0).unwrap() * x2.powi(2)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{f32, f64};

    #[test]
    fn test_goldsteinprice_optimum() {
        assert!((goldsteinprice(&[0.0_f32, -1.0_f32]) - 3_f32).abs() < f32::EPSILON);
        assert!((goldsteinprice(&[0.0_f64, -1.0_f64]) - 3_f64).abs() < f64::EPSILON);
    }

    #[test]
    #[should_panic]
    fn test_goldsteinprice_param_length() {
        goldsteinprice(&[0.0_f32, -1.0_f32, 0.1_f32]);
    }
}
