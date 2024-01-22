// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Rastrigin test function
//!
//! Defined as
//!
//! `f(x_1, x_2, ..., x_n) = a * n + \sum_{i=1}^{n} \left[ x_i^2 - a * cos(2 * pi * x_i) \right]`
//!
//! where `x_i \in [-5.12, 5.12]` and `a = 10`
//!
//! The global minimum is at `f(x_1, x_2, ..., x_n) = f(0, 0, ..., 0) = 0`.

use num::{Float, FromPrimitive};
use std::f64::consts::PI;
use std::iter::Sum;

/// Rastrigin test function
///
/// Defined as
///
/// `f(x_1, x_2, ..., x_n) = a * n + \sum_{i=1}^{n} \left[ x_i^2 - a * cos(2 * pi * x_i) \right]`
///
/// where `x_i \in [-5.12, 5.12]` and `a = 10`
///
/// The global minimum is at `f(x_1, x_2, ..., x_n) = f(0, 0, ..., 0) = 0`.
pub fn rastrigin<T: Float + FromPrimitive + Sum>(param: &[T]) -> T {
    rastrigin_a(param, T::from_f64(10.0).unwrap())
}

/// Rastrigin test function
///
/// The same as `rastrigin`; however, it allows to set the parameter a.
pub fn rastrigin_a<T: Float + FromPrimitive + Sum>(param: &[T], a: T) -> T {
    a * T::from_usize(param.len()).unwrap()
        + param
            .iter()
            .map(|&x| x.powi(2) - a * (T::from_f64(2.0 * PI).unwrap() * x).cos())
            .sum()
}

mod tests {
    // use super::{rastrigin, rastrigin_a};
    // use std;

    #[test]
    fn test_rastrigin_optimum() {
        assert!(::rastrigin(&[0.0_f32, 0.0_f32]).abs() < ::std::f32::EPSILON);
        assert!(::rastrigin(&[0.0_f64, 0.0_f64]).abs() < ::std::f64::EPSILON);
    }

    #[test]
    fn test_parameter_a() {
        assert!(
            ::rastrigin(&[0.0_f32, 0.0_f32]).abs()
                == ::rastrigin_a(&[0.0_f32, 0.0_f32], 10.0).abs()
        );
    }
}
