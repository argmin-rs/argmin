// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Easom test function
//!
//! Defined as
//!
//! `f(x_1, x_2) = - cos(x_1) * cos(x_2) * exp(-(x_1 - pi)^2 - (x_2 - pi)^2)`
//!
//! where `x_i \in [-100, 100]`.
//!
//! The global minimum is at `f(x_1, x_2) = f(pi, pi) = -1`.

use num::{Float, FromPrimitive};
use std::f64::consts::PI;

/// Easom test function
///
/// Defined as
///
/// `f(x_1, x_2) = - cos(x_1) * cos(x_2) * exp(-(x_1 - pi)^2 - (x_2 - pi)^2)`
///
/// where `x_i \in [-100, 100]`.
///
/// The global minimum is at `f(x_1, x_2) = f(pi, pi) = -1`.
pub fn easom<T: Float + FromPrimitive>(param: &[T]) -> T {
    assert!(param.len() == 2);
    let (x1, x2) = (param[0], param[1]);
    let pi = T::from_f64(PI).unwrap();
    -x1.cos() * x2.cos() * (-(x1 - pi).powi(2) - (x2 - pi).powi(2)).exp()
}

mod tests {
    #[test]
    fn test_easom_optimum() {
        assert!(
            (::easom(&[::std::f32::consts::PI, ::std::f32::consts::PI]) + 1.0_f32).abs()
                < ::std::f32::EPSILON
        );
        assert!(
            (::easom(&[::std::f64::consts::PI, ::std::f64::consts::PI]) + 1.0_f64).abs()
                < ::std::f64::EPSILON
        );
    }

    #[test]
    #[should_panic]
    fn test_easom_param_length() {
        ::easom(&[0.0_f32, -1.0_f32, 0.1_f32]);
    }
}
