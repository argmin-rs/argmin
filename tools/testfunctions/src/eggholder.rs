// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Eggholder test function
//!
//! Defined as
//!
//! `f(x_1, x_2) = -(x_2 + 47) * sin( sqrt( abs( x_2 + x_1/2 + 47 ) ) ) -
//!                x_1 * sin( sqrt( abs( x_1 - (x_2 + 47) ) ) )`
//!
//! where `x_i \in [-512, 512]`.
//!
//! The global minimum is at * `f(x_1, x_2) = f(512, 404.2319) = -959.6407`.

use num::{Float, FromPrimitive};

/// Eggholder test function
///
/// Defined as
///
/// `f(x_1, x_2) = -(x_2 + 47) * sin( sqrt( abs( x_2 + x_1/2 + 47 ) ) ) -
///                x_1 * sin( sqrt( abs( x_1 - (x_2 + 47) ) ) )`
///
/// where `x_i \in [-512, 512]`.
///
/// The global minimum is at * `f(x_1, x_2) = f(512, 404.2319) = -959.6407`.
pub fn eggholder<T: Float + FromPrimitive>(param: &[T]) -> T {
    assert!(param.len() == 2);
    let (x1, x2) = (param[0], param[1]);
    let n47 = T::from_f64(47.0).unwrap();
    -(x2 + n47)
        * (x2 + x1 / T::from_f64(2.0).unwrap() + n47)
            .abs()
            .sqrt()
            .sin()
        - x1 * (x1 - (x2 + n47)).abs().sqrt().sin()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eggholder_optimum() {
        assert!((eggholder(&[512.0_f32, 404.2319_f32]) + 959.6407_f32).abs() < std::f32::EPSILON);
    }

    #[test]
    #[should_panic]
    fn test_eggholder_param_length() {
        eggholder(&[0.0_f32, -1.0_f32, 0.1_f32]);
    }
}
