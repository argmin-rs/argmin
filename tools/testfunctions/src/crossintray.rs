// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Cross-in-tray test function
//!
//! Defined as
//!
//! `f(x_1, x_2) = -0.0001 * ( | sin(x_1)*sin(x_2)*exp(| 100 -
//!                                                      \sqrt{x_1^2+_2^2) / pi |) | + 1)^0.1`
//!
//! where `x_i \in [-10, 10]`.
//!
//! The global minima are at
//!  * `f(x_1, x_2) = f(1.34941, 1.34941) = -2.06261`.
//!  * `f(x_1, x_2) = f(1.34941, -1.34941) = -2.06261`.
//!  * `f(x_1, x_2) = f(-1.34941, 1.34941) = -2.06261`.
//!  * `f(x_1, x_2) = f(-1.34941, -1.34941) = -2.06261`.

use std::f64::consts::PI;

use num::{Float, FromPrimitive};

/// Cross-in-tray test function
///
/// Defined as
///
/// `f(x_1, x_2) = -0.0001 * ( | sin(x_1)*sin(x_2)*exp(| 100 -
///                                                      \sqrt{x_1^2+_2^2) / pi |) | + 1)^0.1`
///
/// where `x_i \in [-10, 10]`.
///
/// The global minima are at
///  * `f(x_1, x_2) = f(1.34941, 1.34941) = -2.06261`.
///  * `f(x_1, x_2) = f(1.34941, -1.34941) = -2.06261`.
///  * `f(x_1, x_2) = f(-1.34941, 1.34941) = -2.06261`.
///  * `f(x_1, x_2) = f(-1.34941, -1.34941) = -2.06261`.
///
/// Note: Even if the input parameters are f32, internal computations will be performed in f64.
pub fn cross_in_tray<T>(param: &[T; 2]) -> T
where
    T: Float + Into<f64> + FromPrimitive,
{
    let x1: f64 = param[0].into();
    let x2: f64 = param[1].into();
    T::from_f64(
        -0.0001
            * ((x1.sin() * x2.sin() * (100.0 - (x1.powi(2) + x2.powi(2)).sqrt() / PI).abs().exp())
                .abs()
                + 1.0)
                .powf(0.1),
    )
    .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f32;

    #[test]
    fn test_cross_in_tray_optimum() {
        // This isnt exactly a great way to test this. The function can only be computed with the
        // use of f64; however, I only have the minimum points available in f32, which is why I use
        // the f32 EPSILONs.
        assert_relative_eq!(
            cross_in_tray(&[1.34941_f64, 1.34941_f64]),
            -2.062611870,
            epsilon = f32::EPSILON.into()
        );
        assert_relative_eq!(
            cross_in_tray(&[1.34941_f64, -1.34941_f64]),
            -2.062611870,
            epsilon = f32::EPSILON.into()
        );
        assert_relative_eq!(
            cross_in_tray(&[-1.34941_f64, 1.34941_f64]),
            -2.062611870,
            epsilon = f32::EPSILON.into()
        );
        assert_relative_eq!(
            cross_in_tray(&[-1.34941_f64, -1.34941_f64]),
            -2.062611870,
            epsilon = f32::EPSILON.into()
        );
        assert_relative_eq!(
            cross_in_tray(&[1.34941_f32, 1.34941_f32]),
            -2.062611870,
            epsilon = f32::EPSILON.into()
        );
        assert_relative_eq!(
            cross_in_tray(&[1.34941_f32, -1.34941_f32]),
            -2.062611870,
            epsilon = f32::EPSILON.into()
        );
        assert_relative_eq!(
            cross_in_tray(&[-1.34941_f32, 1.34941_f32]),
            -2.062611870,
            epsilon = f32::EPSILON.into()
        );
        assert_relative_eq!(
            cross_in_tray(&[-1.34941_f32, -1.34941_f32]),
            -2.062611870,
            epsilon = f32::EPSILON.into()
        );
    }
}
