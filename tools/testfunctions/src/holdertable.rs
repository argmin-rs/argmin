// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Holder table test function
//!
//! Defined as
//!
//! `f(x_1, x_2) = -abs(sin(x_1)*cos(x_2)*exp(abs(1- sqrt(x_1^2+x_2^2)/pi)))`
//!
//! where `x_i \in [-10, 10]`.
//!
//! The global minima are at
//!  * `f(x_1, x_2) = f(8.05502, 9.66459) = -19.2085`.
//!  * `f(x_1, x_2) = f(8.05502, -9.66459) = -19.2085`.
//!  * `f(x_1, x_2) = f(-8.05502, 9.66459) = -19.2085`.
//!  * `f(x_1, x_2) = f(-8.05502, -9.66459) = -19.2085`.

use num::{Float, FromPrimitive};
use std::f64::consts::PI;

/// Holder table test function
///
/// Defined as
///
/// `f(x_1, x_2) = -abs(sin(x_1)*cos(x_2)*exp(abs(1- sqrt(x_1^2+x_2^2)/pi)))`
///
/// where `x_i \in [-10, 10]`.
///
/// The global minima are at
///  * `f(x_1, x_2) = f(8.05502, 9.66459) = -19.2085`.
///  * `f(x_1, x_2) = f(8.05502, -9.66459) = -19.2085`.
///  * `f(x_1, x_2) = f(-8.05502, 9.66459) = -19.2085`.
///  * `f(x_1, x_2) = f(-8.05502, -9.66459) = -19.2085`.
pub fn holder_table<T>(param: &[T; 2]) -> T
where
    T: Float + FromPrimitive,
{
    let [x1, x2] = *param;
    let pi = T::from_f64(PI).unwrap();
    let n1 = T::from_f64(1.0).unwrap();
    -(x1.sin() * x2.cos() * (n1 - (x1.powi(2) + x2.powi(2)).sqrt() / pi).abs().exp()).abs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f32;

    #[test]
    fn test_holder_table_optimum() {
        assert_relative_eq!(
            holder_table(&[8.05502_f32, 9.66459_f32]),
            -19.2085,
            epsilon = f32::EPSILON
        );
        assert_relative_eq!(
            holder_table(&[-8.05502_f32, 9.66459_f32]),
            -19.2085,
            epsilon = f32::EPSILON
        );
        assert_relative_eq!(
            holder_table(&[8.05502_f32, -9.66459_f32]),
            -19.2085,
            epsilon = f32::EPSILON
        );
        assert_relative_eq!(
            holder_table(&[-8.05502_f32, -9.66459_f32]),
            -19.2085,
            epsilon = f32::EPSILON
        );
    }
}
