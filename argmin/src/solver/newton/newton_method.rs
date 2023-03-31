// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::{ArgminFloat, Error, Gradient, Hessian, IterState, Problem, Solver, KV};
use argmin_math::{ArgminDot, ArgminInv, ArgminScaledSub};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use std::default::Default;

/// # Newton's method
///
/// Newton's method iteratively finds the stationary points of a function f by using a second order
/// approximation of f at the current point.
///
/// The stepsize `gamma` can be adjusted with the [`with_gamma`](`Newton::with_gamma`) method. It
/// must be in `(0, 1])` and defaults to `1`.
///
/// ## Requirements on the optimization problem
///
/// The optimization problem is required to implement [`Gradient`] and [`Hessian`].
///
/// ## Reference
///
/// Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(Clone, Copy)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct Newton<F> {
    /// gamma
    gamma: F,
}

impl<F> Newton<F>
where
    F: ArgminFloat,
{
    /// Construct a new instance of [`Newton`]
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::newton::Newton;
    /// let newton: Newton<f64> = Newton::new();
    /// ```
    pub fn new() -> Self {
        Newton { gamma: float!(1.0) }
    }

    /// Set step size gamma
    ///
    /// Gamma must be in `(0, 1]` and defaults to `1`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::newton::Newton;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// let newton: Newton<f64> = Newton::new().with_gamma(0.4)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_gamma(mut self, gamma: F) -> Result<Self, Error> {
        if gamma <= float!(0.0) || gamma > float!(1.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "Newton: gamma must be in  (0, 1]."
            ));
        }
        self.gamma = gamma;
        Ok(self)
    }
}

impl<F> Default for Newton<F>
where
    F: ArgminFloat,
{
    fn default() -> Newton<F> {
        Newton::new()
    }
}

impl<O, P, G, H, R, F> Solver<O, IterState<P, G, (), H, R, F>> for Newton<F>
where
    O: Gradient<Param = P, Gradient = G> + Hessian<Param = P, Hessian = H>,
    P: Clone + ArgminScaledSub<P, F, P>,
    H: ArgminInv<H> + ArgminDot<G, P>,
    F: ArgminFloat,
{
    const NAME: &'static str = "Newton method";

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        mut state: IterState<P, G, (), H, R, F>,
    ) -> Result<(IterState<P, G, (), H, R, F>, Option<KV>), Error> {
        let param = state.take_param().ok_or_else(argmin_error_closure!(
            NotInitialized,
            concat!(
                "`Newton` requires an initial parameter vector. ",
                "Please provide an initial guess via `Executor`s `configure` method."
            )
        ))?;
        let grad = problem.gradient(&param)?;
        let hessian = problem.hessian(&param)?;
        let new_param = param.scaled_sub(&self.gamma, &hessian.inv()?.dot(&grad));
        Ok((state.param(new_param), None))
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

    test_trait_impl!(newton_method, Newton<f64>);

    #[test]
    fn test_new() {
        let solver: Newton<f64> = Newton::new();
        assert_eq!(solver.gamma.to_ne_bytes(), 1.0f64.to_ne_bytes());
    }

    #[test]
    fn test_default() {
        let solver_new: Newton<f64> = Newton::new();
        let solver_def: Newton<f64> = Newton::default();
        assert_eq!(
            solver_new.gamma.to_ne_bytes(),
            solver_def.gamma.to_ne_bytes()
        );
    }

    #[test]
    fn test_with_gamma() {
        for new_gamma in [f64::EPSILON, 0.1, 0.5, 0.9, 1.0] {
            let solver: Newton<f64> = Newton::new().with_gamma(new_gamma).unwrap();
            assert_eq!(solver.gamma.to_ne_bytes(), new_gamma.to_ne_bytes());
        }

        for new_gamma in [1.0 + f64::EPSILON, 2.0, 0.0, -1.0] {
            let res = Newton::new().with_gamma(new_gamma);
            assert_error!(
                res,
                ArgminError,
                "Invalid parameter: \"Newton: gamma must be in  (0, 1].\""
            );
        }
    }

    #[cfg(feature = "_ndarrayl")]
    #[test]
    fn test_next_iter_param_not_initialized() {
        use crate::core::State;
        use ndarray::{Array, Array1, Array2};
        let mut newton: Newton<f64> = Newton::new();

        struct NewtonProblem {}

        impl Gradient for NewtonProblem {
            type Param = Array1<f64>;
            type Gradient = Array1<f64>;

            fn gradient(&self, _p: &Self::Param) -> Result<Self::Gradient, Error> {
                Ok(Array1::from_vec(vec![1.0, 2.0]))
            }
        }

        impl Hessian for NewtonProblem {
            type Param = Array1<f64>;
            type Hessian = Array2<f64>;

            fn hessian(&self, _p: &Self::Param) -> Result<Self::Hessian, Error> {
                Ok(Array::from_shape_vec((2, 2), vec![1.0f64, 0.0, 0.0, 1.0])?)
            }
        }

        let res = newton.next_iter(&mut Problem::new(NewtonProblem {}), IterState::new());
        assert_error!(
            res,
            ArgminError,
            concat!(
                "Not initialized: \"`Newton` requires an initial parameter vector. ",
                "Please provide an initial guess via `Executor`s `configure` method.\""
            )
        );
    }

    #[cfg(feature = "_ndarrayl")]
    #[test]
    fn test_solver() {
        use crate::core::State;
        use ndarray::{Array, Array1, Array2};
        struct Problem {}

        impl Gradient for Problem {
            type Param = Array1<f64>;
            type Gradient = Array1<f64>;

            fn gradient(&self, _p: &Self::Param) -> Result<Self::Gradient, Error> {
                Ok(Array1::from_vec(vec![1.0, 2.0]))
            }
        }

        impl Hessian for Problem {
            type Param = Array1<f64>;
            type Hessian = Array2<f64>;

            fn hessian(&self, _p: &Self::Param) -> Result<Self::Hessian, Error> {
                Ok(Array::from_shape_vec((2, 2), vec![1.0f64, 0.0, 0.0, 1.0])?)
            }
        }

        // Single iteration, starting from [0, 0], gamma = 1
        let problem = Problem {};
        let solver: Newton<f64> = Newton::new();
        let init_param = Array1::from_vec(vec![0.0, 0.0]);

        let param = Executor::new(problem, solver)
            .configure(|config| config.param(init_param).max_iters(1))
            .run()
            .unwrap()
            .state
            .get_best_param()
            .unwrap()
            .clone();
        assert_relative_eq!(param[0], -1.0, epsilon = f64::EPSILON);
        assert_relative_eq!(param[1], -2.0, epsilon = f64::EPSILON);

        // Two iterations, starting from [0, 0], gamma = 1
        let problem = Problem {};
        let solver: Newton<f64> = Newton::new();
        let init_param = Array1::from_vec(vec![0.0, 0.0]);

        let param = Executor::new(problem, solver)
            .configure(|config| config.param(init_param).max_iters(2))
            .run()
            .unwrap()
            .state
            .get_best_param()
            .unwrap()
            .clone();
        assert_relative_eq!(param[0], -2.0, epsilon = f64::EPSILON);
        assert_relative_eq!(param[1], -4.0, epsilon = f64::EPSILON);

        // Single iteration, starting from [0, 0], gamma = 0.5
        let problem = Problem {};
        let solver: Newton<f64> = Newton::new().with_gamma(0.5).unwrap();
        let init_param = Array1::from_vec(vec![0.0, 0.0]);

        let param = Executor::new(problem, solver)
            .configure(|config| config.param(init_param).max_iters(1))
            .run()
            .unwrap()
            .state
            .get_best_param()
            .unwrap()
            .clone();
        assert_relative_eq!(param[0], -0.5, epsilon = f64::EPSILON);
        assert_relative_eq!(param[1], -1.0, epsilon = f64::EPSILON);

        // Two iterations, starting from [0, 0], gamma = 0.5
        let problem = Problem {};
        let solver: Newton<f64> = Newton::new().with_gamma(0.5).unwrap();
        let init_param = Array1::from_vec(vec![0.0, 0.0]);

        let param = Executor::new(problem, solver)
            .configure(|config| config.param(init_param).max_iters(2))
            .run()
            .unwrap()
            .state
            .get_best_param()
            .unwrap()
            .clone();
        assert_relative_eq!(param[0], -1.0, epsilon = f64::EPSILON);
        assert_relative_eq!(param[1], -2.0, epsilon = f64::EPSILON);
    }
}
