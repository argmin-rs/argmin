// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Zero
//!
//! Always returns `0.0`. This is only for performance tests.

/// Zero test function
///
/// Always returns `0.0`. This is only for performance tests.
#[inline(always)]
pub fn zero<T>(_param: &[T]) -> f64 {
    0.0
}

/// Derivative of zero test function
///
/// Always returns a vector with the length of param, full of `0.0`. This is only for performance
/// tests.
#[inline(always)]
pub fn zero_derivative<T>(param: &[T]) -> Vec<f64> {
    vec![0.0; param.len()]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std;

    #[test]
    fn test_zero() {
        assert!(zero(&[0.0_f64, 0.0_f64]).abs() < std::f64::EPSILON);
    }
}
