// Copyright 2018-2022 argmin developers
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
//! <https://en.wikipedia.org/wiki/Brent%27s_method>
//!

/// Implementation of Brent's optimization method,
/// see <https://en.wikipedia.org/wiki/Brent%27s_method>
use crate::core::{
    ArgminFloat, Error, IterState, OpWrapper, Operator, Solver, State, TerminationReason, KV,
};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Error to be thrown if Brent is initialized with improper parameters.
#[derive(Debug, Error)]
pub enum BrentError {
    /// f(min) and f(max) must have different signs
    #[error("Brent error: f(min) and f(max) must have different signs.")]
    WrongSign,
}

/// Brent's method
///
/// A root-finding algorithm combining the bisection method, the secant method
/// and inverse quadratic interpolation. It has the reliability of bisection
/// but it can be as quick as some of the less-reliable methods.
///
/// # References:
/// <https://en.wikipedia.org/wiki/Brent%27s_method>
#[derive(Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct Brent<F> {
    /// required relative accuracy
    tol: F,
    /// left or right boundary of current interval
    a: F,
    /// currently proposed best guess
    b: F,
    /// left or right boundary of current interval
    c: F,
    /// helper variable
    d: F,
    /// another helper variable
    e: F,
    /// function value at `a`
    fa: F,
    /// function value at `b`
    fb: F,
    /// function value at `c`
    fc: F,
}

impl<F: ArgminFloat> Brent<F> {
    /// Constructor
    /// The values `min` and `max` must bracketing the root of the function.
    /// The parameter `tol` specifies the relative error to be targeted.
    pub fn new(min: F, max: F, tol: F) -> Brent<F> {
        Brent {
            tol,
            a: min,
            b: max,
            c: max,
            d: F::nan(),
            e: F::nan(),
            fa: F::nan(),
            fb: F::nan(),
            fc: F::nan(),
        }
    }
}

impl<O, F> Solver<O, IterState<F, (), (), (), F>> for Brent<F>
where
    O: Operator<Param = F, Output = F>,
    F: ArgminFloat,
{
    const NAME: &'static str = "Brent";

    fn init(
        &mut self,
        op: &mut OpWrapper<O>,
        // Brent maintains its own state
        state: IterState<F, (), (), (), F>,
    ) -> Result<(IterState<F, (), (), (), F>, Option<KV>), Error> {
        self.fa = op.apply(&self.a)?;
        self.fb = op.apply(&self.b)?;
        if self.fa * self.fb > F::from_f64(0.0).unwrap() {
            return Err(BrentError::WrongSign.into());
        }
        self.fc = self.fb;
        Ok((state.param(self.b).cost(self.fb.abs()), None))
    }

    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        // Brent maintains its own state
        state: IterState<F, (), (), (), F>,
    ) -> Result<(IterState<F, (), (), (), F>, Option<KV>), Error> {
        if (self.fb > F::from_f64(0.0).unwrap() && self.fc > F::from_f64(0.0).unwrap())
            || self.fb < F::from_f64(0.0).unwrap() && self.fc < F::from_f64(0.0).unwrap()
        {
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
        let eff_tol = F::from_f64(2.0).unwrap() * F::epsilon() * self.b.abs()
            + F::from_f64(0.5).unwrap() * self.tol;
        let mid = F::from_f64(0.5).unwrap() * (self.c - self.b);
        if mid.abs() <= eff_tol || self.fb == F::from_f64(0.0).unwrap() {
            return Ok((
                state
                    .termination_reason(TerminationReason::TargetPrecisionReached)
                    .param(self.b)
                    .cost(self.fb.abs()),
                None,
            ));
        }
        if self.e.abs() >= eff_tol && self.fa.abs() > self.fb.abs() {
            let s = self.fb / self.fa;
            let (mut p, mut q) = if self.a == self.c {
                (
                    F::from_f64(2.0).unwrap() * mid * s,
                    F::from_f64(1.0).unwrap() - s,
                )
            } else {
                let q = self.fa / self.fc;
                let r = self.fb / self.fc;
                (
                    s * (F::from_f64(2.0).unwrap() * mid * q * (q - r)
                        - (self.b - self.a) * (r - F::from_f64(1.0).unwrap())),
                    (q - F::from_f64(1.0).unwrap())
                        * (r - F::from_f64(1.0).unwrap())
                        * (s - F::from_f64(1.0).unwrap()),
                )
            };
            if p > F::from_f64(0.0).unwrap() {
                q = -q;
            }
            p = p.abs();
            let min1 = F::from_f64(3.0).unwrap() * mid * q - (eff_tol * q).abs();
            let min2 = (self.e * q).abs();
            if F::from_f64(2.0).unwrap() * p < if min1 < min2 { min1 } else { min2 } {
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
            self.b = self.b + self.d;
        } else {
            self.b = self.b
                + if mid >= F::from_f64(0.0).unwrap() {
                    eff_tol.abs()
                } else {
                    -eff_tol.abs()
                };
        }

        self.fb = op.apply(&self.b)?;
        Ok((state.param(self.b).cost(self.fb.abs()), None))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_trait_impl;

    test_trait_impl!(brent, Brent<f64>);
}
