// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Styblinski-Tang test function
//!
//! Defined as
//!
//! `f(x_1, x_2, ..., x_n) = 1/2 * \sum_{i=1}^{n} \left[ x_i^4 - 16 * x_i^2 + 5 * x_i \right]`
//!
//! where `x_i \in [-5, 5]`.
//!
//! The global minimum is at `f(x_1, x_2, ..., x_n) = f(-2.903534, -2.903534, ..., -2.903534) =
//! -39.16616*n`.

use num::{Float, FromPrimitive};
use std::iter::Sum;

/// Styblinski-Tang test function
///
/// Defined as
///
/// `f(x_1, x_2, ..., x_n) = 1/2 * \sum_{i=1}^{n} \left[ x_i^4 - 16 * x_i^2 + 5 * x_i \right]`
///
/// where `x_i \in [-5, 5]`.
///
/// The global minimum is at `f(x_1, x_2, ..., x_n) = f(-2.903534, -2.903534, ..., -2.903534) =
/// -39.16616*n`.
pub fn styblinski_tang<T: Float + FromPrimitive + Sum>(param: &[T]) -> T {
    T::from_f64(0.5).unwrap()
        * param
            .iter()
            .map(|x| {
                x.powi(4) - T::from_f64(16.0).unwrap() * x.powi(2) + T::from_f64(5.0).unwrap() * *x
            })
            .sum()
}

mod tests {
    #[test]
    fn test_styblinski_tang_optimum() {
        assert!(
            (::styblinski_tang(&[-2.903534_f32, -2.903534_f32, -2.903534_f32]) + 117.49849_f32)
                .abs()
                < ::std::f32::EPSILON
        );
    }
}
