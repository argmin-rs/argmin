// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Zero
//!
//! Always returns `0.0`. This is only for performance tests.

use num::{Float, FromPrimitive};

/// Zero test function
///
/// Always returns `0.0`. This is only for performance tests.
pub fn zero<T>(_param: &[T]) -> T
where
    T: Float + FromPrimitive,
{
    T::from_f64(0.0).unwrap()
}

/// Derivative of zero test function
///
/// Always returns a vector with the length of param, full of `0.0`. This is only for performance
/// tests.
pub fn zero_derivative<T>(param: &[T]) -> Vec<T>
where
    T: Float + FromPrimitive,
{
    vec![T::from_f64(0.0).unwrap(); param.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero() {
        assert_eq!(
            zero(&[0.0_f64, 0.0_f64]).to_ne_bytes(),
            0.0_f64.to_ne_bytes()
        );
        assert_eq!(
            zero(&[0.0_f32, 0.0_f32]).to_ne_bytes(),
            0.0_f32.to_ne_bytes()
        );
    }

    #[test]
    fn test_zero_derivative() {
        zero_derivative(&[0.0_f64, 0.0, 23.0, 28.0])
            .iter()
            .map(|x| assert_eq!(x.to_ne_bytes(), 0.0_f64.to_ne_bytes()))
            .count();

        zero_derivative(&[0.0_f32, 0.0, 23.0, 28.0])
            .iter()
            .map(|x| assert_eq!(x.to_ne_bytes(), 0.0_f32.to_ne_bytes()))
            .count();
    }
}
