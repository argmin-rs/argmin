// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::{
    ArgminFloat, Error, IterState, Jacobian, Operator, Problem, Solver, State, TerminationReason,
    TerminationStatus, KV,
};
use argmin_math::{ArgminDot, ArgminInv, ArgminL2Norm, ArgminMul, ArgminSub, ArgminTranspose};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// # Gauss-Newton method
///
/// The Gauss-Newton method is used to solve non-linear least squares problems.
///
/// Requires an initial parameter vector.
///
/// ## Requirements on the optimization problem
///
/// The optimization problem is required to implement [`Operator`] and [`Jacobian`].
///
/// ## Reference
///
/// Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct GaussNewton<F> {
    /// gamma
    gamma: F,
    /// Tolerance for the stopping criterion based on cost difference
    tol: F,
}

impl<F: ArgminFloat> GaussNewton<F> {
    /// Construct a new instance of [`GaussNewton`].
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::gaussnewton::GaussNewton;
    /// let gauss_newton: GaussNewton<f64> = GaussNewton::new();
    /// ```
    pub fn new() -> Self {
        GaussNewton {
            gamma: float!(1.0),
            tol: F::epsilon().sqrt(),
        }
    }

    /// Set step width gamma.
    ///
    /// Gamma must be within `(0, 1]`. Defaults to `1.0`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::gaussnewton::GaussNewton;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// let gauss_newton = GaussNewton::new().with_gamma(0.5f64)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_gamma(mut self, gamma: F) -> Result<Self, Error> {
        if gamma <= float!(0.0) || gamma > float!(1.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "Gauss-Newton: gamma must be in  (0, 1]."
            ));
        }
        self.gamma = gamma;
        Ok(self)
    }

    /// Set tolerance for the stopping criterion based on cost difference.
    ///
    /// Tolerance must be larger than zero and defaults to `sqrt(EPSILON)`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::gaussnewton::GaussNewton;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// let gauss_newton = GaussNewton::new().with_tolerance(1e-4f64)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_tolerance(mut self, tol: F) -> Result<Self, Error> {
        if tol <= float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "Gauss-Newton: tol must be positive."
            ));
        }
        self.tol = tol;
        Ok(self)
    }
}

impl<F: ArgminFloat> Default for GaussNewton<F> {
    fn default() -> GaussNewton<F> {
        GaussNewton::new()
    }
}

impl<O, F, P, J, U> Solver<O, IterState<P, (), J, (), F>> for GaussNewton<F>
where
    O: Operator<Param = P, Output = U> + Jacobian<Param = P, Jacobian = J>,
    P: Clone + ArgminSub<P, P> + ArgminMul<F, P>,
    U: ArgminL2Norm<F>,
    J: Clone
        + ArgminTranspose<J>
        + ArgminInv<J>
        + ArgminDot<J, J>
        + ArgminDot<U, P>
        + ArgminDot<P, P>,
    F: ArgminFloat,
{
    const NAME: &'static str = "Gauss-Newton method";

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        state: IterState<P, (), J, (), F>,
    ) -> Result<(IterState<P, (), J, (), F>, Option<KV>), Error> {
        let param = state.get_param().ok_or_else(argmin_error_closure!(
            NotInitialized,
            concat!(
                "`GaussNewton` requires an initial parameter vector. ",
                "Please provide an initial guess via `Executor`s `configure` method."
            )
        ))?;
        let residuals = problem.apply(param)?;
        let jacobian = problem.jacobian(param)?;

        let p = jacobian
            .clone()
            .t()
            .dot(&jacobian)
            .inv()?
            .dot(&jacobian.t().dot(&residuals));

        let new_param = param.sub(&p.mul(&self.gamma));

        Ok((state.param(new_param).cost(residuals.l2_norm()), None))
    }

    fn terminate(&mut self, state: &IterState<P, (), J, (), F>) -> TerminationStatus {
        if (state.get_prev_cost() - state.get_cost()).abs() < self.tol {
            return TerminationStatus::Terminated(TerminationReason::SolverConverged);
        }
        TerminationStatus::NotTerminated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::ArgminError;
    #[cfg(feature = "_ndarrayl")]
    use crate::core::Executor;
    use crate::test_trait_impl;
    #[cfg(feature = "_ndarrayl")]
    use approx::assert_relative_eq;

    test_trait_impl!(gauss_newton_method, GaussNewton<f64>);

    #[test]
    fn test_new() {
        let GaussNewton { tol: t, gamma: g } = GaussNewton::<f64>::new();

        assert_eq!(g.to_ne_bytes(), (1.0f64).to_ne_bytes());
        assert_eq!(t.to_ne_bytes(), f64::EPSILON.sqrt().to_ne_bytes());
    }

    #[test]
    fn test_tolerance() {
        let tol1: f64 = 1e-4;

        let GaussNewton { tol: t, .. } = GaussNewton::new().with_tolerance(tol1).unwrap();

        assert_eq!(t.to_ne_bytes(), tol1.to_ne_bytes());
    }

    #[test]
    fn test_tolerance_error() {
        let tol = -2.0;
        let error = GaussNewton::new().with_tolerance(tol);
        assert_error!(
            error,
            ArgminError,
            "Invalid parameter: \"Gauss-Newton: tol must be positive.\""
        );
    }

    #[test]
    fn test_gamma() {
        let gamma: f64 = 0.5;

        let GaussNewton { gamma: g, .. } = GaussNewton::new().with_gamma(gamma).unwrap();

        assert_eq!(g.to_ne_bytes(), gamma.to_ne_bytes());
    }

    #[test]
    fn test_gamma_errors() {
        let gamma = -0.5;
        let error = GaussNewton::new().with_gamma(gamma);
        assert_error!(
            error,
            ArgminError,
            "Invalid parameter: \"Gauss-Newton: gamma must be in  (0, 1].\""
        );

        let gamma = 0.0;
        let error = GaussNewton::new().with_gamma(gamma);
        assert_error!(
            error,
            ArgminError,
            "Invalid parameter: \"Gauss-Newton: gamma must be in  (0, 1].\""
        );

        let gamma = 2.0;
        let error = GaussNewton::new().with_gamma(gamma);
        assert_error!(
            error,
            ArgminError,
            "Invalid parameter: \"Gauss-Newton: gamma must be in  (0, 1].\""
        );
    }

    #[cfg(feature = "_ndarrayl")]
    #[test]
    fn test_next_iter_param_not_initialized() {
        use ndarray::{Array, Array1, Array2};

        struct TestProblem {}

        impl Operator for TestProblem {
            type Param = Array1<f64>;
            type Output = Array1<f64>;

            fn apply(&self, _p: &Self::Param) -> Result<Self::Output, Error> {
                Ok(Array1::from_vec(vec![0.5, 2.0]))
            }
        }

        impl Jacobian for TestProblem {
            type Param = Array1<f64>;
            type Jacobian = Array2<f64>;

            fn jacobian(&self, _p: &Self::Param) -> Result<Self::Jacobian, Error> {
                Ok(Array::from_shape_vec((2, 2), vec![1f64, 2.0, 3.0, 4.0])?)
            }
        }

        let mut gn = GaussNewton::<f64>::new();
        let res = gn.next_iter(&mut Problem::new(TestProblem {}), IterState::new());
        assert_error!(
            res,
            ArgminError,
            concat!(
                "Not initialized: \"`GaussNewton` requires an initial parameter vector. ",
                "Please provide an initial guess via `Executor`s `configure` method.\""
            )
        );
    }

    #[cfg(feature = "_ndarrayl")]
    #[test]
    fn test_solver() {
        use crate::core::State;
        use ndarray::{Array, Array1, Array2};
        use std::cell::RefCell;

        struct Problem {
            counter: RefCell<usize>,
        }

        impl Operator for Problem {
            type Param = Array1<f64>;
            type Output = Array1<f64>;

            fn apply(&self, _p: &Self::Param) -> Result<Self::Output, Error> {
                if *self.counter.borrow() == 0 {
                    let mut c = self.counter.borrow_mut();
                    *c += 1;
                    Ok(Array1::from_vec(vec![0.5, 2.0]))
                } else {
                    Ok(Array1::from_vec(vec![0.3, 1.0]))
                }
            }
        }

        impl Jacobian for Problem {
            type Param = Array1<f64>;
            type Jacobian = Array2<f64>;

            fn jacobian(&self, _p: &Self::Param) -> Result<Self::Jacobian, Error> {
                Ok(Array::from_shape_vec((2, 2), vec![1f64, 2.0, 3.0, 4.0])?)
            }
        }

        // Single iteration, starting from [0, 0], gamma = 1
        let problem = Problem {
            counter: RefCell::new(0),
        };
        let solver: GaussNewton<f64> = GaussNewton::new();
        let init_param = Array1::from_vec(vec![0.0, 0.0]);

        let param = Executor::new(problem, solver)
            .configure(|config| config.param(init_param).max_iters(1))
            .run()
            .unwrap()
            .state
            .get_best_param()
            .unwrap()
            .clone();
        assert_relative_eq!(param[0], -1.0, epsilon = f64::EPSILON.sqrt());
        assert_relative_eq!(param[1], 0.25, epsilon = f64::EPSILON.sqrt());

        // Two iterations, starting from [0, 0], gamma = 1
        let problem = Problem {
            counter: RefCell::new(0),
        };
        let solver: GaussNewton<f64> = GaussNewton::new();
        let init_param = Array1::from_vec(vec![0.0, 0.0]);

        let param = Executor::new(problem, solver)
            .configure(|config| config.param(init_param).max_iters(2))
            .run()
            .unwrap()
            .state
            .get_best_param()
            .unwrap()
            .clone();
        assert_relative_eq!(param[0], -1.4, epsilon = f64::EPSILON.sqrt());
        assert_relative_eq!(param[1], 0.3, epsilon = f64::EPSILON.sqrt());

        // Single iteration, starting from [0, 0], gamma = 0.5
        let problem = Problem {
            counter: RefCell::new(0),
        };
        let solver: GaussNewton<f64> = GaussNewton::new().with_gamma(0.5).unwrap();
        let init_param = Array1::from_vec(vec![0.0, 0.0]);

        let param = Executor::new(problem, solver)
            .configure(|config| config.param(init_param).max_iters(1))
            .run()
            .unwrap()
            .state
            .get_best_param()
            .unwrap()
            .clone();
        assert_relative_eq!(param[0], -0.5, epsilon = f64::EPSILON.sqrt());
        assert_relative_eq!(param[1], 0.125, epsilon = f64::EPSILON.sqrt());

        // Two iterations, starting from [0, 0], gamma = 0.5
        let problem = Problem {
            counter: RefCell::new(0),
        };
        let solver: GaussNewton<f64> = GaussNewton::new().with_gamma(0.5).unwrap();
        let init_param = Array1::from_vec(vec![0.0, 0.0]);

        let param = Executor::new(problem, solver)
            .configure(|config| config.param(init_param).max_iters(2))
            .run()
            .unwrap()
            .state
            .get_best_param()
            .unwrap()
            .clone();
        assert_relative_eq!(param[0], -0.7, epsilon = f64::EPSILON.sqrt());
        assert_relative_eq!(param[1], 0.15, epsilon = f64::EPSILON.sqrt());
    }
}
