// Copyright 2018-2020 argmin developers
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
pub fn picheny<T: Float + FromPrimitive>(param: &[T]) -> T {
    assert!(param.len() == 2);
    let (x1, x2) = (
        T::from_f64(4.0).unwrap() * param[0] - T::from_f64(2.0).unwrap(),
        T::from_f64(4.0).unwrap() * param[1] - T::from_f64(2.0).unwrap(),
    );
    T::from_f64(1.0 / 2.427).unwrap()
        * (((T::from_f64(1.0).unwrap()
            + (x1 + x2 + T::from_f64(1.0).unwrap()).powi(2)
                * (T::from_f64(19.0).unwrap() - T::from_f64(14.0).unwrap() * (x1 + x2)
                    + T::from_f64(3.0).unwrap() * (x1.powi(2) + x2.powi(2))
                    + T::from_f64(6.0).unwrap() * x1 * x2))
            * (T::from_f64(30.0).unwrap()
                + (T::from_f64(2.0).unwrap() * x1 - T::from_f64(3.0).unwrap() * x2).powi(2)
                    * (T::from_f64(18.0).unwrap() - T::from_f64(32.0).unwrap() * x1
                        + T::from_f64(12.0).unwrap() * x1.powi(2)
                        + T::from_f64(48.0).unwrap() * x2
                        - T::from_f64(36.0).unwrap() * x1 * x2
                        + T::from_f64(27.0).unwrap() * x2.powi(2))))
        .log10()
            - T::from_f64(8.693).unwrap())
}

mod tests {
    #[test]
    fn test_picheny_optimum() {
        assert!((::picheny(&[0.5_f32, 0.25_f32]) + 3.3851993182_f32).abs() < ::std::f32::EPSILON);
        assert!(
            (::picheny(&[0.5_f64, 0.25_f64]) + 3.3851993182036826_f64).abs() < ::std::f64::EPSILON
        );
    }

    #[test]
    #[should_panic]
    fn test_picheny_param_length() {
        ::picheny(&[0.0_f32, -1.0_f32, 0.1_f32]);
    }
}
