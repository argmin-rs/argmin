// Copyright 2018-2020 argmin developers
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
pub fn cross_in_tray(param: &[f64]) -> f64 {
    assert!(param.len() == 2);
    let (x1, x2) = (param[0], param[1]);
    // let pi = T::from_f64(PI).unwrap();
    // T::from_f64(-0.0001).unwrap()
    //     * ((x1.sin() * x2.sin()
    //         * (T::from_f64(100.0).unwrap() - (x1.powi(2) + x2.powi(2)).sqrt() / pi)
    //             .abs()
    //             .exp())
    //         .abs() + T::from_f64(1.0).unwrap())
    //         .powf(T::from_f64(0.1).unwrap())
    -0.0001
        * ((x1.sin() * x2.sin() * (100.0 - (x1.powi(2) + x2.powi(2)).sqrt() / PI).abs().exp())
            .abs()
            + 1.0)
            .powf(0.1)
}

mod tests {
    #[test]
    fn test_cross_in_tray_optimum() {
        // This isnt exactly a great way to test this. The function can only be computed with the
        // use of f64; however, I only have the minimum points available in f32, which is why I use
        // the f32 EPSILONs.
        assert!(
            (::cross_in_tray(&[1.34941_f64, 1.34941_f64]) + 2.062611870).abs()
                < ::std::f32::EPSILON.into()
        );
        assert!(
            (::cross_in_tray(&[1.34941_f64, -1.34941_f64]) + 2.062611870).abs()
                < ::std::f32::EPSILON.into()
        );
        assert!(
            (::cross_in_tray(&[-1.34941_f64, 1.34941_f64]) + 2.062611870).abs()
                < ::std::f32::EPSILON.into()
        );
        assert!(
            (::cross_in_tray(&[-1.34941_f64, -1.34941_f64]) + 2.062611870).abs()
                < ::std::f32::EPSILON.into()
        );
    }

    #[test]
    #[should_panic]
    fn test_cross_in_tray_param_length() {
        ::cross_in_tray(&[0.0, -1.0, 0.1]);
    }
}
