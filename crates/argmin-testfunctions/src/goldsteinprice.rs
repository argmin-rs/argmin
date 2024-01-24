// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Goldstein-Price test function
//!
//! Defined as
//!
//! `f(x_1, x_2) = [1 + (x_1 + x_2 + 1)^2 * (19 - 14*x_2 + + 3*x_1^2 - 14*x_2 6*x_1*x_2 + 3*x_2^2)]
//!                * [30 + (2*x_1 - 3*x_2)^2(18 - 32 * x_1 + 12* x_1^2 + 48 * x_2 -
//!                   36 * x_1 * x_2 + 27 * x_2^2) ]`
//!
//! where `x_i \in [-2, 2]`.
//!
//! The global minimum is at `f(x_1, x_2) = f(0, -1) = 3`.

use num::{Float, FromPrimitive};

/// Goldstein-Price test function
///
/// Defined as
///
/// `f(x_1, x_2) = [1 + (x_1 + x_2 + 1)^2 * (19 - 14*x_2 + 3*x_1^2 - 14*x_2 6*x_1*x_2 + 3*x_2^2)]
///                * [30 + (2*x_1 - 3*x_2)^2(18 - 32 * x_1 + 12* x_1^2 + 48 * x_2 -
///                   36 * x_1 * x_2 + 27 * x_2^2) ]`
///
/// where `x_i \in [-2, 2]`.
///
/// The global minimum is at `f(x_1, x_2) = f(0, -1) = 3`.
pub fn goldsteinprice<T>(param: &[T; 2]) -> T
where
    T: Float + FromPrimitive,
{
    let [x1, x2] = *param;
    let n1 = T::from_f64(1.0).unwrap();
    let n2 = T::from_f64(2.0).unwrap();
    let n3 = T::from_f64(3.0).unwrap();
    let n6 = T::from_f64(6.0).unwrap();
    let n12 = T::from_f64(12.0).unwrap();
    let n14 = T::from_f64(14.0).unwrap();
    let n18 = T::from_f64(18.0).unwrap();
    let n19 = T::from_f64(19.0).unwrap();
    let n27 = T::from_f64(27.0).unwrap();
    let n30 = T::from_f64(30.0).unwrap();
    let n32 = T::from_f64(32.0).unwrap();
    let n36 = T::from_f64(36.0).unwrap();
    let n48 = T::from_f64(48.0).unwrap();
    (n1 + (x1 + x2 + n1).powi(2)
        * (n19 - n14 * (x1 + x2) + n3 * (x1.powi(2) + x2.powi(2)) + n6 * x1 * x2))
        * (n30
            + (n2 * x1 - n3 * x2).powi(2)
                * (n18 - n32 * x1 + n12 * x1.powi(2) + n48 * x2 - n36 * x1 * x2 + n27 * x2.powi(2)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::{f32, f64};

    #[test]
    fn test_goldsteinprice_optimum() {
        assert_relative_eq!(
            goldsteinprice(&[0.0_f32, -1.0_f32]),
            3_f32,
            epsilon = f32::EPSILON
        );
        assert_relative_eq!(
            goldsteinprice(&[0.0_f64, -1.0_f64]),
            3_f64,
            epsilon = f64::EPSILON
        );
    }
}
