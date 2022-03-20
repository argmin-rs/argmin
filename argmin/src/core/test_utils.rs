// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::{
    CostFunction, Error, Gradient, Hessian, IterState, Jacobian, Operator, Problem, Solver, KV,
};
use crate::solver::simulatedannealing::Anneal;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// Pseudo problem useful for testing
///
/// Implements [`CostFunction`], [`Operator`], [`Gradient`], [`Jacobian`], [`Hessian`], and
/// [`Anneal`].
#[derive(Clone, Default, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct TestProblem {}

impl TestProblem {
    /// Create an instance of `TestProblem`.
    ///
    /// # Example
    ///
    /// ```
    /// use argmin::core::test_utils::TestProblem;
    ///
    /// let problem = TestProblem::new();
    /// # assert_eq!(problem, TestProblem {});
    /// ```
    #[allow(dead_code)]
    pub fn new() -> Self {
        TestProblem {}
    }
}

impl Operator for TestProblem {
    type Param = Vec<f64>;
    type Output = Vec<f64>;

    /// Returns a clone of parameter `p`.
    ///
    /// # Example
    ///
    /// ```
    /// use argmin::core::test_utils::TestProblem;
    /// use argmin::core::Operator;
    /// # use argmin::core::Error;
    ///
    /// # fn main() -> Result<(), Error> {
    /// let problem = TestProblem::new();
    ///
    /// let param = vec![1.0, 2.0];
    ///
    /// let res = problem.apply(&param)?;
    /// # assert_eq!(res, param);
    /// # Ok(())
    /// # }
    /// ```
    fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        Ok(p.clone())
    }
}

impl CostFunction for TestProblem {
    type Param = Vec<f64>;
    type Output = f64;

    /// Returns `1.0f64`.
    ///
    /// # Example
    ///
    /// ```
    /// use argmin::core::test_utils::TestProblem;
    /// use argmin::core::CostFunction;
    /// # use argmin::core::Error;
    ///
    /// # fn main() -> Result<(), Error> {
    /// let problem = TestProblem::new();
    ///
    /// let param = vec![1.0, 2.0];
    ///
    /// let res = problem.cost(&param)?;
    /// # assert_eq!(res, 1.0f64);
    /// # Ok(())
    /// # }
    /// ```
    fn cost(&self, _p: &Self::Param) -> Result<Self::Output, Error> {
        Ok(1.0f64)
    }
}

impl Gradient for TestProblem {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    /// Returns a clone of parameter `p`.
    ///
    /// # Example
    ///
    /// ```
    /// use argmin::core::test_utils::TestProblem;
    /// use argmin::core::Gradient;
    /// # use argmin::core::Error;
    ///
    /// # fn main() -> Result<(), Error> {
    /// let problem = TestProblem::new();
    ///
    /// let param = vec![1.0, 2.0];
    ///
    /// let res = problem.gradient(&param)?;
    /// # assert_eq!(res, param);
    /// # Ok(())
    /// # }
    /// ```
    fn gradient(&self, p: &Self::Param) -> Result<Self::Param, Error> {
        Ok(p.clone())
    }
}

impl Hessian for TestProblem {
    type Param = Vec<f64>;
    type Hessian = Vec<Vec<f64>>;

    /// Returns `vec![p, p]`.
    ///
    /// # Example
    ///
    /// ```
    /// use argmin::core::test_utils::TestProblem;
    /// use argmin::core::Hessian;
    /// # use argmin::core::Error;
    ///
    /// # fn main() -> Result<(), Error> {
    /// let problem = TestProblem::new();
    ///
    /// let param = vec![1.0, 2.0];
    ///
    /// let res = problem.hessian(&param)?;
    /// # assert_eq!(res, vec![param.clone(), param.clone()]);
    /// # Ok(())
    /// # }
    /// ```
    fn hessian(&self, p: &Self::Param) -> Result<Self::Hessian, Error> {
        Ok(vec![p.clone(), p.clone()])
    }
}

impl Jacobian for TestProblem {
    type Param = Vec<f64>;
    type Jacobian = Vec<Vec<f64>>;

    /// Returns `vec![p, p]`.
    ///
    /// # Example
    ///
    /// ```
    /// use argmin::core::test_utils::TestProblem;
    /// use argmin::core::Jacobian;
    /// # use argmin::core::Error;
    ///
    /// # fn main() -> Result<(), Error> {
    /// let problem = TestProblem::new();
    ///
    /// let param = vec![1.0, 2.0];
    ///
    /// let res = problem.jacobian(&param)?;
    /// # assert_eq!(res, vec![param.clone(), param.clone()]);
    /// # Ok(())
    /// # }
    /// ```
    fn jacobian(&self, p: &Self::Param) -> Result<Self::Jacobian, Error> {
        Ok(vec![p.clone(), p.clone()])
    }
}

impl Anneal for TestProblem {
    type Param = Vec<f64>;
    type Output = Vec<f64>;
    type Float = f64;

    /// Returns a clone of parameter `p`.
    ///
    /// # Example
    ///
    /// ```
    /// use argmin::core::test_utils::TestProblem;
    /// use argmin::solver::simulatedannealing::Anneal;
    /// # use argmin::core::Error;
    ///
    /// # fn main() -> Result<(), Error> {
    /// let problem = TestProblem::new();
    ///
    /// let param = vec![1.0, 2.0];
    ///
    /// let res = problem.anneal(&param, 1.0)?;
    /// # assert_eq!(res, param);
    /// # Ok(())
    /// # }
    /// ```
    fn anneal(&self, p: &Self::Param, _t: Self::Float) -> Result<Self::Output, Error> {
        Ok(p.clone())
    }
}

/// A (non-working) solver useful for testing
///
/// Implements the [`Solver`] trait.
#[derive(Clone, Default, Eq, PartialEq, Debug)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct TestSolver {}

impl TestSolver {
    /// Create an instance of `TestSolver`.
    ///
    /// # Example
    ///
    /// ```
    /// use argmin::core::test_utils::TestSolver;
    ///
    /// let solver = TestSolver::new();
    /// # assert_eq!(solver, TestSolver {});
    /// ```
    pub fn new() -> TestSolver {
        TestSolver {}
    }
}

impl<O> Solver<O, IterState<Vec<f64>, (), (), (), f64>> for TestSolver {
    const NAME: &'static str = "TestSolver";

    fn next_iter(
        &mut self,
        _problem: &mut Problem<O>,
        state: IterState<Vec<f64>, (), (), (), f64>,
    ) -> Result<(IterState<Vec<f64>, (), (), (), f64>, Option<KV>), Error> {
        Ok((state, None))
    }
}
