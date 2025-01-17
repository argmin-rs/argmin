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

/// Error to be thrown if ITP method is initialized with improper parameters.
#[derive(Debug, Error)]
pub enum ItpRootError {
    /// f(min) and f(max) must have different signs
    #[error("ItpRoot error: f(min) and f(max) must have different signs.")]
    WrongSign,
    /// tol must be positive
    #[error("ItpRoot error: tol must be positive.")]
    NegativeTol,
    /// tol must be nonzero
    #[error("ItpRoot error: tol must be nonzero.")]
    ZeroTol,
    /// max must be larger than min
    #[error("ItpRoot error: max must be larger than min.")]
    MinLargerThanMax,
}

/// # ITP method
///
/// A root-finding algorithm, short for "interpolate, truncate, and project",
/// that achieves superlinear convergence while retaining the worst-case
/// performance of the bisection method.
///
/// ## Requirements on the optimization problem
///
/// The optimization problem is required to implement [`CostFunction`].
///
/// ##  Reference
///
/// [ITP Method]: https://en.wikipedia.org/wiki/ITP_Method
/// [An Enhancement of the Bisection Method Average Performance Preserving Minmax Optimality]: https://dl.acm.org/doi/10.1145/3423597
#[derive(Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct ItpRoot<F> {
    /// required relative accuracy
    tol: F,
    /// tuned hyper-parameter 1, controls truncation size
    kappa1: F,
    /// tuned hyper-parameter 2, controls truncation size
    kappa2: F,
    /// tuned hyper-parameter slack variable, controls projection interval size
    n0: F,
    /// left boundary of current interval
    a: F,
    /// right boundary of current interval
    b: F,
    /// function value at `a`
    fa: F,
    /// function value at `b`
    fb: F,
    /// iteration counter, used in the projection step
    j: F,
    /// n_(1/2), a preprocessing variable
    n1o2: F,
    /// a preprocessing variable
    nmax: F,
}

impl<F: ArgminFloat> ItpRoot<F> {
    /// Constructor
    /// The values `min` and `max` must be bracketing the root of the function.
    /// The parameter `tol` specifies the relative error to be targeted.
    /// The values `kappa1` and `kappa2` are hyper-parameters tuning the truncation size.
    /// The parameter `n0` is a hyper-parameter slack variable controlling the projection interval
    /// size.
    pub fn new(min: F, max: F, tol: F, kappa1: F, kappa2: F, n0: F) -> Result<Self, Error> {
        if tol < F::zero() {
            return Err(ItpRootError::NegativeTol.into());
        }
        // This helps ensure the log evaluation (in the init) is stable
        if min > max {
            return Err(ItpRootError::MinLargerThanMax.into());
        }
        // It's important to check this to verify n1o2 doesn't panic
        if tol.is_zero() {
            return Err(ItpRootError::ZeroTol.into());
        }

        Ok(ItpRoot {
            tol,
            kappa1,
            kappa2,
            n0,
            a: min,
            b: max,
            fa: F::nan(),
            fb: F::nan(),
            // Starts at zero, increments in the solver
            j: F::zero(),
            // These get computed in the solver
            n1o2: F::nan(),
            nmax: F::nan(),
        })
    }

    /// Constructor with default hyperparameters
    /// The values `min` and `max` must be bracketing the root of the function.
    /// The parameter `tol` specifies the relative error to be targeted.
    /// kappa1 is defaulted to 0.2 / (max - min).
    /// kappa2 is defaulted to 2.0.
    /// n0 is defaulted to 1.0.
    pub fn from_defaults(min: F, max: F, tol: F) -> Result<Self, Error> {
        Self::new(
            min,
            max,
            tol,
            // kappa1, suggested from paper
            float!(0.2) / (max - min),
            // kappa2
            float!(2.0),
            // n0
            float!(1.0),
        )
    }
}

impl<O, F> Solver<O, IterState<F, (), (), (), (), F>> for ItpRoot<F>
where
    O: CostFunction<Param = F, Output = F>,
    F: ArgminFloat,
{
    fn name(&self) -> &str {
        "ItpRoot"
    }

    fn init(
        &mut self,
        problem: &mut Problem<O>,
        // ItpRoot maintains its own state
        state: IterState<F, (), (), (), (), F>,
    ) -> Result<(IterState<F, (), (), (), (), F>, Option<KV>), Error> {
        self.fa = problem.cost(&self.a)?;
        self.fb = problem.cost(&self.b)?;
        if self.fa * self.fb > float!(0.0) {
            return Err(ItpRootError::WrongSign.into());
        }

        // Preprocessing
        self.n1o2 = ((self.b - self.a) / (float!(2.0) * self.tol)).log2();
        self.nmax = self.n1o2 + self.n0;

        Ok((state.param(self.b).cost(self.fb.abs()), None))
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        // ItpRoot maintains its own state
        state: IterState<F, (), (), (), (), F>,
    ) -> Result<(IterState<F, (), (), (), (), F>, Option<KV>), Error> {
        // Note: the headers here match the steps outlined on Wikipedia, with variable names to
        // match for clarity.

        // Calculating Parameters
        let b_minus_a = self.b - self.a;
        let x1o2 = (self.a + self.b) * float!(0.5);
        let r = self.tol * float!(2.0).powf(self.nmax + self.j) - b_minus_a * float!(0.5);
        let delta = self.kappa1 * b_minus_a.powf(self.kappa2);

        // Interpolation
        let xf = (self.fb * self.a - self.fa * self.b) / (self.fb - self.fa);

        // Truncation
        let bisect_falsi_delta = x1o2 - xf;
        let sigma = bisect_falsi_delta.signum();
        let xt = if delta <= bisect_falsi_delta.abs() {
            xf + sigma * delta
        } else {
            x1o2
        };

        // Projection
        let xitp = if (xt - x1o2).abs() <= r {
            xt
        } else {
            x1o2 - sigma * r
        };

        // Updating Interval
        let fitp = problem.cost(&xitp)?;
        if fitp > F::zero() {
            self.b = xitp;
            self.fb = fitp;
        } else if fitp < F::zero() {
            self.a = xitp;
            self.fa = fitp;
        } else {
            self.a = xitp;
            self.b = xitp;
        }
        self.j = self.j + float!(1.0);

        // Solver loop termination
        if (self.b - self.a) <= (float!(2.0) * self.tol) {
            let sol = (self.a + self.b) * float!(0.5);
            // TODO: This function evaluation serves no purpose other than to serve argmin's cost
            // method on the state. It feels wasteful.
            let f_sol = problem.cost(&sol)?;
            return Ok((
                state
                    .terminate_with(TerminationReason::SolverConverged)
                    .param(sol)
                    .cost(f_sol.abs()),
                None,
            ));
        }

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

    // This polynomial matches what is explored in the Wikipedia example
    #[derive(Clone)]
    struct Polynomial {}

    impl CostFunction for Polynomial {
        type Param = f64;
        type Output = f64;

        fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
            Ok(param.powi(3) - param - 2.0) // x^3 - x - 2
        }
    }

    #[test]
    fn test_itp_negative_tol() {
        let min: f64 = 0.0;
        let max: f64 = 2.0;
        let tol: f64 = -1e-6;

        let result = ItpRoot::from_defaults(min, max, tol);

        // Check if the initialization fails and we get the correct error message
        assert!(result.is_err());
        assert_eq!(
            result.err().unwrap().to_string(),
            "ItpRoot error: tol must be positive."
        );
    }

    #[test]
    fn test_itp_invalid_range() {
        let min: f64 = 2.0;
        let max: f64 = 3.0;
        let tol: f64 = 1e-6;

        let mut solver: ItpRoot<f64> = ItpRoot::from_defaults(min, max, tol).unwrap();
        let mut problem: Problem<Quadratic> = Problem::new(Quadratic {});

        let result: Result<(IterState<f64, (), (), (), (), f64>, Option<KV>), Error> =
            solver.init(&mut problem, IterState::new());

        // Check if the initialization fails and we get the correct error message
        assert!(result.is_err());
        assert_eq!(
            result.err().unwrap().to_string(),
            "ItpRoot error: f(min) and f(max) must have different signs."
        );
    }

    #[test]
    fn test_itp_valid_range() {
        let min: f64 = 0.0;
        let max: f64 = 2.0;
        let tol: f64 = 1e-6;

        let mut solver: ItpRoot<f64> = ItpRoot::from_defaults(min, max, tol).unwrap();
        let mut problem: Problem<Quadratic> = Problem::new(Quadratic {});

        let result: Result<(IterState<f64, (), (), (), (), f64>, Option<KV>), Error> =
            solver.init(&mut problem, IterState::new());

        // Check if the initialization is successful
        assert!(result.is_ok());
    }

    #[test]
    fn test_itp_find_quadratic_root() {
        let min: f64 = 0.0;
        let max: f64 = 2.0;
        let tol: f64 = 1e-6;
        let init_param: f64 = 1.5;

        let solver: ItpRoot<f64> = ItpRoot::from_defaults(min, max, tol).unwrap();
        let problem: Quadratic = Quadratic {};

        let res = Executor::new(problem, solver)
            .configure(|state| state.param(init_param).max_iters(100))
            .run()
            .unwrap();

        // Check if the result is close to the real root
        assert_relative_eq!(res.state.best_param.unwrap(), 1.0, epsilon = tol);
    }

    #[test]
    fn test_itp_find_polynomial_root() {
        let min: f64 = 1.0;
        let max: f64 = 2.0;
        let tol: f64 = 0.0005;
        let kappa1: f64 = 0.1;
        let kappa2: f64 = 2.0;
        let n0: f64 = 1.0;
        let init_param: f64 = 1.5;

        let solver: ItpRoot<f64> = ItpRoot::new(min, max, tol, kappa1, kappa2, n0).unwrap();
        let problem: Polynomial = Polynomial {};

        let res = Executor::new(problem, solver)
            .configure(|state| state.param(init_param).max_iters(100))
            .run()
            .unwrap();

        // Check if the result is close to the real root
        assert_relative_eq!(
            res.state.best_param.unwrap(),
            1.52138301273268,
            epsilon = tol
        );
    }
}
