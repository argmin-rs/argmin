// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::{
    ArgminFloat, CostFunction, Error, IterState, Problem, Solver, State, TerminationReason, KV,
};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Error to be thrown if Brent is initialized with improper parameters.
#[derive(Debug, Error)]
pub enum BrentRootError {
    /// f(min) and f(max) must have different signs
    #[error("BrentRoot error: f(min) and f(max) must have different signs.")]
    WrongSign,
    // tol must be positive
    #[error("BrentRoot error: tol must be positive.")]
    NegativeTol,
}

/// # Brent's method
///
/// A root-finding algorithm combining the bisection method, the secant method
/// and inverse quadratic interpolation. It has the reliability of bisection
/// but it can be as quick as some of the less-reliable methods.
///
/// ## Requirements on the optimization problem
///
/// The optimization problem is required to implement [`CostFunction`].
///
/// ##  Reference
///
/// <https://en.wikipedia.org/wiki/Brent%27s_method>
#[derive(Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct BrentRoot<F> {
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

impl<F: ArgminFloat> BrentRoot<F> {
    /// Constructor
    /// The values `min` and `max` must bracketing the root of the function.
    /// The parameter `tol` specifies the relative error to be targeted.
    pub fn new(min: F, max: F, tol: F) -> Self {
        BrentRoot {
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

impl<O, F> Solver<O, IterState<F, (), (), (), (), F>> for BrentRoot<F>
where
    O: CostFunction<Param = F, Output = F>,
    F: ArgminFloat,
{
    fn name(&self) -> &str {
        "BrentRoot"
    }

    fn init(
        &mut self,
        problem: &mut Problem<O>,
        // BrentRoot maintains its own state
        state: IterState<F, (), (), (), (), F>,
    ) -> Result<(IterState<F, (), (), (), (), F>, Option<KV>), Error> {
        self.fa = problem.cost(&self.a)?;
        self.fb = problem.cost(&self.b)?;
        if self.fa * self.fb > float!(0.0) {
            return Err(BrentRootError::WrongSign.into());
        }
        if self.tol < F::zero() {
            return Err(BrentRootError::NegativeTol.into());
        }
        self.fc = self.fb;
        Ok((state.param(self.b).cost(self.fb.abs()), None))
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        // BrentRoot maintains its own state
        state: IterState<F, (), (), (), (), F>,
    ) -> Result<(IterState<F, (), (), (), (), F>, Option<KV>), Error> {
        if (self.fb > float!(0.0) && self.fc > float!(0.0))
            || self.fb < float!(0.0) && self.fc < float!(0.0)
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
        let eff_tol = float!(2.0) * F::epsilon() * self.b.abs() + float!(0.5) * self.tol;
        let mid = float!(0.5) * (self.c - self.b);
        if mid.abs() <= eff_tol || self.fb == float!(0.0) {
            return Ok((
                state
                    .terminate_with(TerminationReason::SolverConverged)
                    .param(self.b)
                    .cost(self.fb.abs()),
                None,
            ));
        }
        if self.e.abs() >= eff_tol && self.fa.abs() > self.fb.abs() {
            let s = self.fb / self.fa;
            let (mut p, mut q) = if self.a == self.c {
                (float!(2.0) * mid * s, float!(1.0) - s)
            } else {
                let q = self.fa / self.fc;
                let r = self.fb / self.fc;
                (
                    s * (float!(2.0) * mid * q * (q - r) - (self.b - self.a) * (r - float!(1.0))),
                    (q - float!(1.0)) * (r - float!(1.0)) * (s - float!(1.0)),
                )
            };
            if p > float!(0.0) {
                q = -q;
            }
            p = p.abs();
            let min1 = float!(3.0) * mid * q - (eff_tol * q).abs();
            let min2 = (self.e * q).abs();
            if float!(2.0) * p < if min1 < min2 { min1 } else { min2 } {
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
                + if mid >= float!(0.0) {
                    eff_tol.abs()
                } else {
                    -eff_tol.abs()
                };
        }

        self.fb = problem.cost(&self.b)?;
        Ok((state.param(self.b).cost(self.fb.abs()), None))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Executor;
    use approx::assert_relative_eq;

    #[derive(Clone)]
    struct Quadratic {}

    impl CostFunction for Quadratic {
        type Param = f64;
        type Output = f64;

        fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
            Ok(param.powi(2) - 1.0) // x^2 - 1
        }
    }

    #[test]
    fn test_brent_negative_tol() {
        let min: f64 = 0.0;
        let max: f64 = 2.0;
        let tol: f64 = -1e-6;

        let mut solver: BrentRoot<f64> = BrentRoot::new(min, max, tol);
        let mut problem: Problem<Quadratic> = Problem::new(Quadratic {});

        let result: Result<(IterState<f64, (), (), (), (), f64>, Option<KV>), Error> =
            solver.init(&mut problem, IterState::new());

        // Check if the initialization fails and we get the correct error message
        assert!(result.is_err());
        assert_eq!(
            result.err().unwrap().to_string(),
            "BrentRoot error: tol must be positive."
        );
    }

    #[test]
    fn test_brent_invalid_range() {
        let min: f64 = 2.0;
        let max: f64 = 3.0;
        let tol: f64 = 1e-6;

        let mut solver: BrentRoot<f64> = BrentRoot::new(min, max, tol);
        let mut problem: Problem<Quadratic> = Problem::new(Quadratic {});

        let result: Result<(IterState<f64, (), (), (), (), f64>, Option<KV>), Error> =
            solver.init(&mut problem, IterState::new());

        // Check if the initialization fails and we get the correct error message
        assert!(result.is_err());
        assert_eq!(
            result.err().unwrap().to_string(),
            "BrentRoot error: f(min) and f(max) must have different signs."
        );
    }

    #[test]
    fn test_brent_valid_range() {
        let min: f64 = 0.0;
        let max: f64 = 2.0;
        let tol: f64 = 1e-6;

        let mut solver: BrentRoot<f64> = BrentRoot::new(min, max, tol);
        let mut problem: Problem<Quadratic> = Problem::new(Quadratic {});

        let result: Result<(IterState<f64, (), (), (), (), f64>, Option<KV>), Error> =
            solver.init(&mut problem, IterState::new());

        // Check if the initialization is successful
        assert!(result.is_ok());
    }

    #[test]
    fn test_brent_find_root() {
        let min: f64 = 0.0;
        let max: f64 = 2.0;
        let tol: f64 = 1e-6;
        let init_param: f64 = 1.5;

        let solver: BrentRoot<f64> = BrentRoot::new(min, max, tol);
        let problem: Quadratic = Quadratic {};

        let res = Executor::new(problem, solver)
            .configure(|state| state.param(init_param).max_iters(100))
            .run()
            .unwrap();

        // Check if the result is close to the real root
        assert_relative_eq!(res.state.best_param.unwrap(), 1.0, epsilon = tol);
    }

    #[test]
    fn test_brent_symmetry() {
        let min: f64 = 0.0;
        let max: f64 = 2.0;
        let tol: f64 = 1e-6;
        let init_param: f64 = 1.5;

        let problem: Quadratic = Quadratic {};

        // First run with [min, max] interval
        let solver1: BrentRoot<f64> = BrentRoot::new(min, max, tol);
        let res1 = Executor::new(problem.clone(), solver1)
            .configure(|state| state.param(init_param).max_iters(100))
            .run()
            .unwrap();

        // Second run with [max, min] interval (swapped inputs)
        let solver2: BrentRoot<f64> = BrentRoot::new(max, min, tol);
        let res2 = Executor::new(problem, solver2)
            .configure(|state| state.param(init_param).max_iters(100))
            .run()
            .unwrap();

        // Check if the results are the same
        assert_relative_eq!(
            res1.state.param.unwrap(),
            res2.state.param.unwrap(),
            epsilon = tol,
        );

        // Check if the number of iterations is the same
        assert_eq!(res1.state.get_iter(), res2.state.get_iter());
    }
}
