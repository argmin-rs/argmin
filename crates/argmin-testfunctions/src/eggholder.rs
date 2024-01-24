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
pub fn eggholder<T>(param: &[T; 2]) -> T
where
    T: Float + FromPrimitive,
{
    let [x1, x2] = *param;
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
    use approx::assert_relative_eq;
    use std::{f32, f64};

    #[test]
    fn test_eggholder_optimum() {
        assert_relative_eq!(
            eggholder(&[512.0_f32, 404.2319_f32]),
            -959.6407_f32,
            epsilon = f32::EPSILON
        );
        assert_relative_eq!(
            eggholder(&[512.0_f64, 404.2319_f64]),
            -959.6406627106155_f64,
            epsilon = f64::EPSILON
        );
    }
}
