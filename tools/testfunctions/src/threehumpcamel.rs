// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Three-hump camel test function
//!
//! Defined as
//!
//! `f(x_1, x_2) = 2*x_1^2 - 1.05*x_1^4 + x_1^6/6 + x_1*x_2 + x_2^2`
//!
//! where `x_i \in [-5, 5]`.
//!
//! The global minimum is at `f(x_1, x_2) = f(0, 0) = 0`.

use num::{Float, FromPrimitive};

/// Three-hump camel test function
///
/// Defined as
///
/// `f(x_1, x_2) = 2*x_1^2 - 1.05*x_1^4 + x_1^6/6 + x_1*x_2 + x_2^2`
///
/// where `x_i \in [-5, 5]`.
///
/// The global minimum is at `f(x_1, x_2) = f(0, 0) = 0`.
pub fn threehumpcamel<T: Float + FromPrimitive>(param: &[T]) -> T {
    assert!(param.len() == 2);
    let (x1, x2) = (param[0], param[1]);
    T::from_f64(2.0).unwrap() * x1.powi(2) - T::from_f64(1.05).unwrap() * x1.powi(4)
        + x1.powi(6) / T::from_f64(6.0).unwrap()
        + x1 * x2
        + x2.powi(2)
}

mod tests {
    #[test]
    fn test_threehumpcamel_optimum() {
        assert!((::threehumpcamel(&[0.0_f32, 0.0_f32])).abs() < ::std::f32::EPSILON);
        assert!((::threehumpcamel(&[0.0_f64, 0.0_f64])).abs() < ::std::f64::EPSILON);
    }

    #[test]
    #[should_panic]
    fn test_threehumpcamel_param_length() {
        ::threehumpcamel(&[0.0_f32, -1.0_f32, 0.1_f32]);
    }
}
