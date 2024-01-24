// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::{
    ArgminFloat, CostFunction, Error, Executor, Gradient, IterState, Jacobian, LineSearch,
    Operator, OptimizationResult, Problem, Solver, TerminationReason, TerminationStatus, KV,
};
use argmin_math::{ArgminDot, ArgminInv, ArgminL2Norm, ArgminMul, ArgminTranspose};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// # Gauss-Newton method with line search
///
/// Gauss-Newton method where an appropriate step length is obtained by a line search.
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
pub struct GaussNewtonLS<L, F> {
    /// linesearch
    linesearch: L,
    /// Tolerance for the stopping criterion based on cost difference
    tol: F,
}

impl<L, F: ArgminFloat> GaussNewtonLS<L, F> {
    /// Construct a new instance of [`GaussNewtonLS`].
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::gaussnewton::GaussNewtonLS;
    /// # let linesearch = ();
    /// let gauss_newton_ls: GaussNewtonLS<_, f64> = GaussNewtonLS::new(linesearch);
    /// ```
    pub fn new(linesearch: L) -> Self {
        GaussNewtonLS {
            linesearch,
            tol: F::epsilon().sqrt(),
        }
    }

    /// Set tolerance for the stopping criterion based on cost difference.
    ///
    /// Tolerance must be larger than zero and defaults to `sqrt(EPSILON)`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::gaussnewton::GaussNewtonLS;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// # let linesearch = ();
    /// let gauss_newton_ls = GaussNewtonLS::new(linesearch).with_tolerance(1e-4f64)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_tolerance(mut self, tol: F) -> Result<Self, Error> {
        if tol <= float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "Gauss-Newton-Linesearch: tol must be positive."
            ));
        }
        self.tol = tol;
        Ok(self)
    }
}

impl<O, L, F, P, G, J, U, R> Solver<O, IterState<P, G, J, (), R, F>> for GaussNewtonLS<L, F>
where
    O: Operator<Param = P, Output = U> + Jacobian<Param = P, Jacobian = J>,
    P: Clone + ArgminMul<F, P>,
    G: Clone,
    U: ArgminL2Norm<F>,
    J: Clone
        + ArgminTranspose<J>
        + ArgminInv<J>
        + ArgminDot<J, J>
        + ArgminDot<G, P>
        + ArgminDot<U, G>,
    L: Clone + LineSearch<P, F> + Solver<LineSearchProblem<O, F>, IterState<P, G, (), (), R, F>>,
    F: ArgminFloat,
    R: Clone,
{
    const NAME: &'static str = "Gauss-Newton method with line search";

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        mut state: IterState<P, G, J, (), R, F>,
    ) -> Result<(IterState<P, G, J, (), R, F>, Option<KV>), Error> {
        let param = state.take_param().ok_or_else(argmin_error_closure!(
            NotInitialized,
            concat!(
                "`GaussNewtonLS` requires an initial parameter vector. ",
                "Please provide an initial guess via `Executor`s `configure` method."
            )
        ))?;
        let residuals = problem.apply(&param)?;
        let jacobian = problem.jacobian(&param)?;
        let jacobian_t = jacobian.clone().t();
        let grad = jacobian_t.dot(&residuals);

        let p: P = jacobian_t.dot(&jacobian).inv()?.dot(&grad);

        self.linesearch.search_direction(p.mul(&(float!(-1.0))));

        // perform linesearch
        let OptimizationResult {
            problem: mut line_problem,
            state: mut linesearch_state,
            ..
        } = Executor::new(
            LineSearchProblem::new(problem.take_problem().ok_or_else(argmin_error_closure!(
                PotentialBug,
                "`GaussNewtonLS`: Failed to take `problem` for line search"
            ))?),
            self.linesearch.clone(),
        )
        .configure(|config| config.param(param).gradient(grad).cost(residuals.l2_norm()))
        .ctrlc(false)
        .run()?;

        // Here we cannot use `consume_problem` because the problem we need is hidden inside a
        // `LineSearchProblem` hidden inside a `Problem`. Therefore we have to split this in two
        // separate tasks: first getting the problem, then dealing with the function counts.
        problem.problem = Some(
            line_problem
                .take_problem()
                .ok_or_else(argmin_error_closure!(
                    PotentialBug,
                    "`GaussNewtonLS`: Failed to take `problem` from line search"
                ))?
                .problem,
        );
        problem.consume_func_counts(line_problem);

        Ok((
            state
                .param(
                    linesearch_state
                        .take_param()
                        .ok_or_else(argmin_error_closure!(
                            PotentialBug,
                            "`GaussNewtonLS`: Failed to take `param` from line search state"
                        ))?,
                )
                .cost(linesearch_state.get_cost()),
            None,
        ))
    }

    fn terminate(&mut self, state: &IterState<P, G, J, (), R, F>) -> TerminationStatus {
        if (state.get_prev_cost() - state.get_cost()).abs() < self.tol {
            return TerminationStatus::Terminated(TerminationReason::SolverConverged);
        }
        TerminationStatus::NotTerminated
    }
}

#[doc(hidden)]
#[derive(Clone, Default)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
struct LineSearchProblem<O, F> {
    problem: O,
    _phantom: std::marker::PhantomData<F>,
}

impl<O, F> LineSearchProblem<O, F> {
    /// Construct a new [`LineSearchProblem`]
    fn new(operator: O) -> Self {
        LineSearchProblem {
            problem: operator,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<O, P, F> CostFunction for LineSearchProblem<O, F>
where
    O: Operator<Param = P, Output = P>,
    P: Clone + ArgminL2Norm<F>,
    F: ArgminFloat,
{
    type Param = P;
    type Output = F;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        Ok(self.problem.apply(p)?.l2_norm())
    }
}

impl<O, P, J, F> Gradient for LineSearchProblem<O, F>
where
    O: Operator<Param = P, Output = P> + Jacobian<Param = P, Jacobian = J>,
    P: Clone,
    J: ArgminTranspose<J> + ArgminDot<P, P>,
{
    type Param = P;
    type Gradient = P;

    fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
        Ok(self.problem.jacobian(p)?.t().dot(&self.problem.apply(p)?))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::let_unit_value)]

    use super::*;
    use crate::core::ArgminError;
    #[cfg(feature = "_ndarrayl")]
    use crate::core::{IterState, State};
    use crate::solver::linesearch::{condition::ArmijoCondition, BacktrackingLineSearch};
    use crate::{assert_error, test_trait_impl};
    #[cfg(feature = "_ndarrayl")]
    use approx::assert_relative_eq;

    test_trait_impl!(
        gauss_newton_linesearch_method,
        GaussNewtonLS<BacktrackingLineSearch<Vec<f64>, Vec<f64>, ArmijoCondition<f64>, f64>, f64>
    );

    #[test]
    fn test_new() {
        #[derive(Eq, PartialEq, Debug)]
        struct MyLinesearch {}

        let GaussNewtonLS {
            linesearch: ls,
            tol: t,
        } = GaussNewtonLS::<_, f64>::new(MyLinesearch {});

        assert_eq!(ls, MyLinesearch {});
        assert_eq!(t.to_ne_bytes(), f64::EPSILON.sqrt().to_ne_bytes());
    }

    #[test]
    fn test_tolerance() {
        let tol1: f64 = 1e-4;

        let linesearch = ();
        let GaussNewtonLS { tol: t1, .. } =
            GaussNewtonLS::new(linesearch).with_tolerance(tol1).unwrap();

        assert_eq!(t1.to_ne_bytes(), tol1.to_ne_bytes());
    }

    #[test]
    fn test_tolerance_error_when_negative() {
        let tol = -2.0;
        let error = GaussNewtonLS::new(()).with_tolerance(tol);
        assert_error!(
            error,
            ArgminError,
            "Invalid parameter: \"Gauss-Newton-Linesearch: tol must be positive.\""
        );
    }

    #[test]
    fn test_tolerance_error_when_zero() {
        let tol = 0.0;
        let error = GaussNewtonLS::new(()).with_tolerance(tol);
        assert_error!(
            error,
            ArgminError,
            "Invalid parameter: \"Gauss-Newton-Linesearch: tol must be positive.\""
        );
    }

    #[cfg(feature = "_ndarrayl")]
    #[test]
    fn test_line_search_sub_problem() {
        use ndarray::{Array, Array1, Array2};

        struct TestProblem {}

        impl Operator for TestProblem {
            type Param = Array1<f64>;
            type Output = Array1<f64>;

            fn apply(&self, _p: &Self::Param) -> Result<Self::Output, Error> {
                Ok(Array1::from_vec(vec![0.5, 2.0]))
            }
        }

        impl Gradient for TestProblem {
            type Param = Array1<f64>;
            type Gradient = Array1<f64>;

            fn gradient(&self, _p: &Self::Param) -> Result<Self::Gradient, Error> {
                Ok(Array1::from_vec(vec![1.5, 3.0]))
            }
        }

        impl Jacobian for TestProblem {
            type Param = Array1<f64>;
            type Jacobian = Array2<f64>;

            fn jacobian(&self, _p: &Self::Param) -> Result<Self::Jacobian, Error> {
                Ok(Array::from_shape_vec((2, 2), vec![1f64, 2.0, 3.0, 4.0])?)
            }
        }

        let lsp: LineSearchProblem<_, f64> = LineSearchProblem::new(TestProblem {});

        let res = lsp.cost(&Array1::from_vec(vec![])).unwrap();
        assert_relative_eq!(
            res,
            (0.5f64.powi(2) + 2.0f64.powi(2)).sqrt(),
            epsilon = f64::EPSILON
        );

        let res = lsp.gradient(&Array1::from_vec(vec![])).unwrap();
        assert_relative_eq!(res[0], 1.0 * 0.5 + 3.0 * 2.0, epsilon = f64::EPSILON);
        assert_relative_eq!(res[1], 2.0 * 0.5 + 4.0 * 2.0, epsilon = f64::EPSILON);
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

        let linesearch: BacktrackingLineSearch<
            Array1<f64>,
            Array1<f64>,
            ArmijoCondition<f64>,
            f64,
        > = BacktrackingLineSearch::new(ArmijoCondition::new(0.2).unwrap());
        let mut gnls = GaussNewtonLS::<_, f64>::new(linesearch);
        let res = gnls.next_iter(&mut Problem::new(TestProblem {}), IterState::new());
        assert_error!(
            res,
            ArgminError,
            concat!(
                "Not initialized: \"`GaussNewtonLS` requires an initial parameter vector. ",
                "Please provide an initial guess via `Executor`s `configure` method.\""
            )
        );
    }

    #[cfg(feature = "_ndarrayl")]
    #[test]
    fn test_next_iter_regression() {
        use ndarray::{Array, Array1, Array2};
        use std::cell::RefCell;

        struct MyProblem {
            counter: RefCell<usize>,
        }

        impl Operator for MyProblem {
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

        impl Gradient for MyProblem {
            type Param = Array1<f64>;
            type Gradient = Array1<f64>;

            fn gradient(&self, _p: &Self::Param) -> Result<Self::Gradient, Error> {
                Ok(Array1::from_vec(vec![3.2, 1.1]))
            }
        }

        impl Jacobian for MyProblem {
            type Param = Array1<f64>;
            type Jacobian = Array2<f64>;

            fn jacobian(&self, _p: &Self::Param) -> Result<Self::Jacobian, Error> {
                Ok(Array::from_shape_vec((2, 2), vec![1f64, 2.0, 3.0, 4.0])?)
            }
        }

        let problem = MyProblem {
            counter: RefCell::new(0),
        };

        let linesearch: BacktrackingLineSearch<
            Array1<f64>,
            Array1<f64>,
            ArmijoCondition<f64>,
            f64,
        > = BacktrackingLineSearch::new(ArmijoCondition::new(0.2).unwrap());
        let mut gnls = GaussNewtonLS::<_, f64>::new(linesearch);
        let state = IterState::new()
            .param(Array1::from_vec(vec![1.0, 2.0]))
            .jacobian(Array::from_shape_vec((2, 2), vec![1f64, 2.0, 3.0, 4.0]).unwrap());
        let mut problem = Problem::new(problem);
        let (mut state, kv) = gnls.next_iter(&mut problem, state).unwrap();
        state.update();

        assert!(kv.is_none());

        assert_relative_eq!(
            state.param.as_ref().unwrap()[0],
            7.105427357601002e-15,
            epsilon = f64::EPSILON
        );
        assert_relative_eq!(
            state.param.as_ref().unwrap()[1],
            2.25f64,
            epsilon = f64::EPSILON
        );
        assert_relative_eq!(
            state.best_param.as_ref().unwrap()[0],
            7.105427357601002e-15,
            epsilon = f64::EPSILON
        );
        assert_relative_eq!(
            state.best_param.as_ref().unwrap()[1],
            2.25f64,
            epsilon = f64::EPSILON
        );
        assert_relative_eq!(state.cost, 1.044030650891055f64, epsilon = f64::EPSILON);
        assert!(!state.prev_cost.is_finite());
        assert_relative_eq!(
            state.best_cost,
            1.044030650891055f64,
            epsilon = f64::EPSILON
        );
    }
}
