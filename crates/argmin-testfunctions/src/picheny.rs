// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Picheny test function
//!
//! Variation of the Goldstein-Price test function.
//!
//! Defined as
//!
//! `f(x_1, x_2) = (1/2.427) * log([1 + (\bar{x}_1 + \bar{x}_2 + 1)^2 * (19 - 14*\bar{x}_2 +
//!                3*\bar{x}_1^2 - 14*\bar{x}_2 6*\bar{x}_1*\bar{x}_2 + 3*\bar{x}_2^2)]
//!                * [30 + (2*\bar{x}_1 - 3*\bar{x}_2)^2(18 - 32 * \bar{x}_1 + 12* \bar{x}_1^2 +
//!                48 * \bar{x}_2 - 36 * \bar{x}_1 * \bar{x}_2 + 27 * \bar{x}_2^2) ] - 8.693)`
//!
//! where `\bar{x}_i = 4*x_i - 2` and `x_i \in [0, 1]`.
//!
//! The global minimum is at `f(x_1, x_2) = f(0.5, 0.25) = 3.3851993182036826`.

use num::{Float, FromPrimitive};

/// Picheny test function
///
/// Variation of the Goldstein-Price test function.
///
/// Defined as
///
/// `f(x_1, x_2) = (1/2.427) * log([1 + (\bar{x}_1 + \bar{x}_2 + 1)^2 * (19 - 14*\bar{x}_2 +
///                3*\bar{x}_1^2 - 14*\bar{x}_2 6*\bar{x}_1*\bar{x}_2 + 3*\bar{x}_2^2)]
///                * [30 + (2*\bar{x}_1 - 3*\bar{x}_2)^2(18 - 32 * \bar{x}_1 + 12* \bar{x}_1^2 +
///                48 * \bar{x}_2 - 36 * \bar{x}_1 * \bar{x}_2 + 27 * \bar{x}_2^2) ] - 8.693)`
///
/// where `\bar{x}_i = 4*x_i - 2` and `x_i \in [0, 1]`.
///
/// The global minimum is at `f(x_1, x_2) = f(0.5, 0.25) = 3.3851993182036826`.
pub fn picheny<T>(param: &[T; 2]) -> T
where
    T: Float + FromPrimitive,
{
    let n1 = T::from_f64(1.0).unwrap();
    let n2 = T::from_f64(2.0).unwrap();
    let n3 = T::from_f64(3.0).unwrap();
    let n4 = T::from_f64(4.0).unwrap();
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

    let [x1, x2] = param.map(|x| n4 * x - n2);

    T::from_f64(1.0 / 2.427).unwrap()
        * (((n1
            + (x1 + x2 + n1).powi(2)
                * (n19 - n14 * (x1 + x2) + n3 * (x1.powi(2) + x2.powi(2)) + n6 * x1 * x2))
            * (n30
                + (n2 * x1 - n3 * x2).powi(2)
                    * (n18 - n32 * x1 + n12 * x1.powi(2) + n48 * x2 - n36 * x1 * x2
                        + n27 * x2.powi(2))))
        .log10()
            - T::from_f64(8.693).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::{f32, f64};

    #[test]
    fn test_picheny_optimum() {
        assert_relative_eq!(
            picheny(&[0.5_f32, 0.25_f32]),
            -3.3851993182,
            epsilon = f32::EPSILON
        );
        assert_relative_eq!(
            picheny(&[0.5_f64, 0.25_f64]),
            -3.3851993182036826,
            epsilon = f64::EPSILON
        );
    }
}
