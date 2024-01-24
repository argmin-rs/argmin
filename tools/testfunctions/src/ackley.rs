// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Ackley test function
//!
//! Defined as
//!
//! `f(x_1, x_2, ..., x_n) = - a * exp( -b \sqrt{\frac{1}{d}\sum_{i=1}^n x_i^2 ) -
//! exp( \frac{1}{d} cos(c * x_i) ) + a + exp(1)`
//!
//! where `x_i \in [-32.768, 32.768]` and usually `a = 10`, `b = 0.2` and `c = 2*pi`
//!
//! The global minimum is at `f(x_1, x_2, ..., x_n) = f(0, 0, ..., 0) = 0`.

use num::{Float, FromPrimitive};
use std::f64::consts::PI;
use std::iter::Sum;

/// Ackley test function
///
/// Defined as
///
/// `f(x_1, x_2, ..., x_n) = - a * exp( -b \sqrt{\frac{1}{d}\sum_{i=1}^n x_i^2 ) -
/// exp( \frac{1}{d} cos(c * x_i) ) + a + exp(1)`
///
/// where `x_i \in [-32.768, 32.768]` and usually `a = 10`, `b = 0.2` and `c = 2*pi`
///
/// The global minimum is at `f(x_1, x_2, ..., x_n) = f(0, 0, ..., 0) = 0`.
pub fn ackley<T: Float + FromPrimitive + Sum>(param: &[T]) -> T {
    ackley_param(
        param,
        T::from_f64(20.0).unwrap(),
        T::from_f64(0.2).unwrap(),
        T::from_f64(2.0 * PI).unwrap(),
    )
}

/// Ackley test function
///
/// The same as `ackley`; however, it allows to set the parameters a, b and c.
pub fn ackley_param<T: Float + FromPrimitive + Sum>(param: &[T], a: T, b: T, c: T) -> T {
    let num1 = T::from_f64(1.0).unwrap();
    let n = T::from_usize(param.len()).unwrap();
    -a * (-b * ((num1 / n) * param.iter().map(|x| x.powi(2)).sum()).sqrt()).exp()
        - ((num1 / n) * param.iter().map(|x| (c * *x).cos()).sum()).exp()
        + a
        + num1.exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{f32, f64};

    #[test]
    fn test_ackley_optimum() {
        // There seem to be numerical problems which is why the epsilons are multiplied with a
        // factor. Not sure if this is acceptable...
        assert!(ackley(&[0.0_f32, 0.0_f32, 0.0_f32]).abs() < f32::EPSILON * 10_f32);
        assert!(ackley(&[0.0_f64, 0.0_f64, 0.0_f64]).abs() < f64::EPSILON * 3_f64);
    }

    #[test]
    fn test_parameters() {
        assert!(
            ackley(&[0.0_f64, 0.0_f64, 0.0_f64]).abs()
                == ackley_param(
                    &[0.0_f64, 0.0_f64, 0.0_f64],
                    20.0,
                    0.2,
                    2.0 * f64::consts::PI
                )
                .abs()
        );
    }
}
