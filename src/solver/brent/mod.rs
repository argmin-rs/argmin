// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Brent's method
//!
//! A root-finding algorithm combining the bisection method, the secant method
//! and inverse quadratic interpolation. It has the reliability of bisection
//! but it can be as quick as some of the less-reliable methods.
//!
//! # References:
//!
//! https://en.wikipedia.org/wiki/Brent%27s_method
//!

/// Implementation of Brent's optimization method,
/// see https://en.wikipedia.org/wiki/Brent%27s_method
use crate::prelude::*;
use serde::{Deserialize, Serialize};
use std::f64;
use std::fmt;

/// Error to be thrown if Brent is initialized with improper parameters.
#[derive(Debug)]
pub struct BrentError;

impl fmt::Display for BrentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Brent error: f(min) and f(max) must have different signs."
        )
    }
}

impl std::error::Error for BrentError {
    fn description(&self) -> &str {
        "Brent requires a min and max value bracketing the root of the function."
    }
}

/// Brent's method
///
/// A root-finding algorithm combining the bisection method, the secant method
/// and inverse quadratic interpolation. It has the reliability of bisection
/// but it can be as quick as some of the less-reliable methods.
///
/// # References:
/// https://en.wikipedia.org/wiki/Brent%27s_method
#[derive(Clone, Serialize, Deserialize)]
pub struct Brent {
    /// required relative accuracy
    tol: f64,
    /// left or right boundary of current interval
    a: f64,
    /// currently proposed best guess
    b: f64,
    /// left or right boundary of current interval
    c: f64,
    /// helper variable
    d: f64,
    /// helper variable 
    e: f64,
    /// function value at `a`
    fa: f64,
    /// function value at `b`
    fb: f64,
    /// function value at `c`
    fc: f64,
}

impl Brent {
    /// Constructor
    /// The values `min` and `max` must bracketing the root of the function.
    /// The parameter `tol` specifies the relative error to be targeted.
    pub fn new(min: f64, max: f64, tol: f64) -> Brent {
        Brent {
            tol: tol,
            a: min,
            b: max,
            c: max,
            d: f64::NAN,
            e: f64::NAN,
            fa: f64::NAN,
            fb: f64::NAN,
            fc: f64::NAN,
        }
    }
}

impl<O> Solver<O> for Brent
where
    O: ArgminOp<Param = f64, Output = f64>,
{
    const NAME: &'static str = "Brent";

    fn init(
        &mut self,
        op: &mut OpWrapper<O>,
        // Brent maintains its own state
        _state: &IterState<O>,
    ) -> Result<Option<ArgminIterData<O>>, Error> {
        self.fa = op.apply(&self.a)?;
        self.fb = op.apply(&self.b)?;
        if self.fa * self.fb > 0.0 {
            return Err(Error::from_boxed_compat(Box::new(BrentError)));
        }
        self.fc = self.fb;
        Ok(Some(
            ArgminIterData::new().param(self.b).cost(self.fb.abs()),
        ))
    }

    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        // Brent maintains its own state
        _state: &IterState<O>,
    ) -> Result<ArgminIterData<O>, Error> {
        if (self.fb > 0.0 && self.fc > 0.0) || self.fb < 0.0 && self.fc < 0.0 {
            self.c = self.a;
            self.fc = self.fa;
            self.d = self.b - self.a;
            self.e = self.d;
        }
        if self.fc.abs() < self.fb.abs() {
            self.a = self.b;
            self.b = self.c;
            self.c = self.a;
            self.fa = self.fb;
            self.fb = self.fc;
            self.fc = self.fa;
        }
        // effective tolerance is double machine precision plus half tolerance as given.
        let eff_tol = 2.0 * f64::EPSILON * self.b.abs() + 0.5 * self.tol;
        let mid = 0.5 * (self.c - self.b);
        if mid.abs() <= eff_tol || self.fb == 0.0 {
            return Ok(ArgminIterData::new()
                .termination_reason(TerminationReason::TargetPrecisionReached)
                .param(self.b)
                .cost(self.fb.abs()));
        }
        if self.e.abs() >= eff_tol && self.fa.abs() > self.fb.abs() {
            let s = self.fb / self.fa;
            let (mut p, mut q) = if self.a == self.c {
                (2.0 * mid * s, 1.0 - s)
            } else {
                let q = self.fa / self.fc;
                let r = self.fb / self.fc;
                (
                    s * (2.0 * mid * q * (q - r) - (self.b - self.a) * (r - 1.0)),
                    (q - 1.0) * (r - 1.0) * (s - 1.0),
                )
            };
            if p > 0.0 {
                q = -q;
            }
            p = p.abs();
            let min1 = 3.0 * mid * q - (eff_tol * q).abs();
            let min2 = (self.e * q).abs();
            if 2.0 * p < if min1 < min2 { min1 } else { min2 } {
                self.e = self.d;
                self.d = p / q;
            } else {
                self.d = mid;
                self.e = self.d;
            };
        } else {
            self.d = mid;
            self.e = self.d;
        };
        self.a = self.b;
        self.fa = self.fb;
        if self.d.abs() > eff_tol {
            self.b += self.d;
        } else {
            self.b += if mid >= 0.0 {
                eff_tol.abs()
            } else {
                -eff_tol.abs()
            };
        }

        self.fb = op.apply(&self.b)?;
        Ok(ArgminIterData::new().param(self.b).cost(self.fb.abs()))
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_trait_impl;

    test_trait_impl!(brent, Brent);
}
