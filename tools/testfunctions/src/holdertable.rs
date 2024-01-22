// Copyright 2018-2020 argmin developers
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
pub fn holder_table<T: Float + FromPrimitive>(param: &[T]) -> T {
    assert!(param.len() == 2);
    let (x1, x2) = (param[0], param[1]);
    let pi = T::from_f64(PI).unwrap();
    -(x1.sin()
        * x2.cos()
        * (T::from_f64(1.0).unwrap() - (x1.powi(2) + x2.powi(2)).sqrt() / pi)
            .abs()
            .exp())
    .abs()
}

mod tests {
    #[test]
    fn test_holder_table_optimum() {
        assert!(
            (::holder_table(&[8.05502_f32, 9.66459_f32]) + 19.2085).abs() < ::std::f32::EPSILON
        );
        assert!(
            (::holder_table(&[8.05502_f32, 9.66459_f32]) + 19.2085).abs() < ::std::f32::EPSILON
        );
        assert!(
            (::holder_table(&[8.05502_f32, 9.66459_f32]) + 19.2085).abs() < ::std::f32::EPSILON
        );
        assert!(
            (::holder_table(&[8.05502_f32, 9.66459_f32]) + 19.2085).abs() < ::std::f32::EPSILON
        );
    }

    #[test]
    #[should_panic]
    fn test_holder_table_param_length() {
        ::holder_table(&[0.0_f32, -1.0_f32, 0.1_f32]);
    }
}
