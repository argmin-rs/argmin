// Copyright 2018-2020 argmin developers
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
pub fn mccorminck<T: Float + FromPrimitive>(param: &[T]) -> T {
    assert!(param.len() == 2);
    let (x1, x2) = (param[0], param[1]);
    (x1 + x2).sin() + (x1 - x2).powi(2) - T::from_f64(1.5).unwrap() * x1
        + T::from_f64(2.5).unwrap() * x2
        + T::from_f64(1.0).unwrap()
}

mod tests {
    #[test]
    fn test_mccorminck_optimum() {
        assert!(
            (::mccorminck(&[-0.54719_f32, -1.54719_f32]) + 1.9132228_f32).abs()
                < ::std::f32::EPSILON
        );
    }

    #[test]
    #[should_panic]
    fn test_mccorminck_param_length() {
        ::mccorminck(&[0.0_f32, -1.0_f32, 0.1_f32]);
    }
}
