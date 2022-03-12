// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::{ArgminError, ArgminFloat, DeserializeOwnedAlias, Error, SerializeAlias};
use std::collections::HashMap;
#[cfg(feature = "serde1")]
use std::default::Default;

/// This wraps an operator and keeps track of how often the cost, gradient and Hessian have been
/// computed. Usually, this is an implementation detail unless a solver is needed within another
/// solver (such as a line search within a gradient descent method).
#[derive(Clone, Debug, Default)]
pub struct Problem<O> {
    /// Operator
    pub problem: Option<O>,
    /// Evaluation counts
    pub counts: HashMap<&'static str, u64>,
}

impl<O> Problem<O> {
    /// Gives access to the stored operator `op` via the closure `func` and keeps track of how many
    /// times the function has been called. The function counts will be passed to observers labelled
    /// with `counts_string`. Per convention, `counts_string` is chosen as `<something>_count`.
    ///
    /// Example
    /// ```
    /// # use argmin::core::{PseudoOperator, Problem, CostFunction};
    /// # let mut problem = Problem::new(PseudoOperator::new());
    /// # let param = vec![1.0f64, 0.0];
    /// let cost = problem.problem("cost_count", |problem| problem.cost(&param));
    /// # assert_eq!(problem.counts["cost_count"], 1)
    /// ```
    ///
    /// This is typically used when designing a trait which optimization problems need to implement
    /// for certain solvers. For instance, for a trait `Anneal` used in Simulated Annealing, one
    /// would write the following to enable the solver to call `.anneal(...)` on an `Problem`
    /// directly:
    ///
    /// ```
    /// # // needs to reimplement Problem because doctests run in dedicated crate where it is not
    /// # // possible to `impl` a type from another crate.
    /// # // Probably somewhat related to https://github.com/rust-lang/rust/issues/50784
    /// # use argmin::core::{Error};
    /// # use std::collections::HashMap;
    /// #
    /// # pub struct Problem<O> {
    /// #     pub problem: Option<O>,
    /// #     pub counts: HashMap<&'static str, u64>,
    /// # }
    /// # impl<O> Problem<O> {
    /// #     pub fn problem<T, F: FnOnce(&O) -> Result<T, Error>>(
    /// #         &mut self,
    /// #         counts_string: &'static str,
    /// #         func: F,
    /// #     ) -> Result<T, Error> {
    /// #         let count = self.counts.entry(counts_string).or_insert(0);
    /// #         *count += 1;
    /// #         func(self.problem.as_ref().unwrap())
    /// #     }
    /// # }
    /// pub trait Anneal {
    ///     type Param;
    ///     type Output;
    ///     type Float;
    ///
    ///     fn anneal(&self, param: &Self::Param, extent: Self::Float) -> Result<Self::Output, Error>;
    /// }
    ///
    /// impl<O: Anneal> Problem<O> {
    ///     pub fn anneal(&mut self, param: &O::Param, extent: O::Float) -> Result<O::Output, Error> {
    ///         self.problem("anneal_count", |problem| problem.anneal(param, extent))
    ///     }
    /// }
    ///
    /// // ...
    ///
    /// # struct TestProblem {}
    /// # impl Anneal for TestProblem {
    /// #     type Param = ();
    /// #     type Output = ();
    /// #     type Float = f64;
    /// #
    /// #     fn anneal(&self, param: &Self::Param, _extent: Self::Float) -> Result<Self::Output, Error> {
    /// #         Ok(())
    /// #     }
    /// # }
    /// # let mut problem = Problem { problem: Some(TestProblem {}), counts: HashMap::new() };
    /// # let param = ();
    /// let new_param = problem.anneal(&param, 1.0f64);
    /// # assert_eq!(problem.counts["anneal_count"], 1)
    /// ```
    ///
    /// Note that this will unfortunately only work inside the `argmin` crate itself due to the fact
    /// that it is not possible to `impl` a type from another crate. Therefore if one implements a
    /// solver outside of argmin, `.problem(...)` has to be called directly as shown in the first
    /// example.
    pub fn problem<T, F: FnOnce(&O) -> Result<T, Error>>(
        &mut self,
        counts_string: &'static str,
        func: F,
    ) -> Result<T, Error> {
        let count = self.counts.entry(counts_string).or_insert(0);
        *count += 1;
        func(self.problem.as_ref().unwrap())
    }
}

impl<O> Problem<O> {
    /// Construct an `Problem` from an operator
    pub fn new(problem: O) -> Self {
        Problem {
            problem: Some(problem),
            counts: HashMap::new(),
        }
    }
}

impl<O> Problem<O> {
    /// Moves the operator out of the struct and replaces it with `None`
    pub fn take_problem(&mut self) -> Option<O> {
        self.problem.take()
    }

    /// Consumes an operator by increasing the function call counts of `self` by the ones in
    /// `other`.
    pub fn consume_problem(&mut self, mut other: Problem<O>) {
        self.problem = Some(other.take_problem().unwrap());
        self.consume_func_counts(other);
    }

    /// Adds function evaluation counts of another operator.
    pub fn consume_func_counts<O2>(&mut self, other: Problem<O2>) {
        for (k, v) in other.counts.iter() {
            let count = self.counts.entry(k).or_insert(0);
            *count += v
        }
    }

    /// Reset the cost function counts to zero.
    #[must_use]
    pub fn reset(mut self) -> Self {
        for (_, v) in self.counts.iter_mut() {
            *v = 0;
        }
        self
    }

    /// Returns the operator `problem` by taking ownership of `self`.
    pub fn get_problem(self) -> O {
        self.problem.unwrap()
    }
}

/// TODO
pub trait Operator {
    /// Type of the parameter vector
    type Param;
    /// Return value of the operator
    type Output;

    /// Applies the operator to parameters
    fn apply(&self, param: &Self::Param) -> Result<Self::Output, Error>;
}

/// TODO
pub trait CostFunction {
    /// Type of the parameter vector
    type Param;
    /// Return value of the cost function
    type Output;

    /// Compute cost function
    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error>;
}

/// TODO
pub trait Gradient {
    /// Type of the parameter vector
    type Param;
    /// Type of the gradient
    type Gradient;

    /// Compute gradient
    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, Error>;
}

/// TODO
pub trait Hessian {
    /// Type of the parameter vector
    type Param;
    /// Type of the Hessian
    type Hessian;

    /// Compute Hessian
    fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, Error>;
}

/// TODO
pub trait Jacobian {
    /// Type of the parameter vector
    type Param;
    /// Type of the Jacobian
    type Jacobian;

    /// Compute Jacobian
    fn jacobian(&self, param: &Self::Param) -> Result<Self::Jacobian, Error>;
}

/// Problems which implement this trait can be used for linear programming solvers
pub trait LinearProgram {
    /// Type of the parameter vector
    type Param: Clone + SerializeAlias + DeserializeOwnedAlias;
    /// Precision of floats
    type Float: ArgminFloat;

    /// TODO c for linear programs
    /// Those three could maybe be merged into a single function; name unclear
    fn c(&self) -> Result<Vec<Self::Float>, Error> {
        Err(ArgminError::NotImplemented {
            text: "Method `c` of LinearProgram trait not implemented!".to_string(),
        }
        .into())
    }

    /// TODO b for linear programs
    fn b(&self) -> Result<Vec<Self::Float>, Error> {
        Err(ArgminError::NotImplemented {
            text: "Method `b` of LinearProgram trait not implemented!".to_string(),
        }
        .into())
    }

    /// TODO A for linear programs
    #[allow(non_snake_case)]
    fn A(&self) -> Result<Vec<Vec<Self::Float>>, Error> {
        Err(ArgminError::NotImplemented {
            text: "Method `A` of LinearProgram trait not implemented!".to_string(),
        }
        .into())
    }
}

impl<O: Operator> Problem<O> {
    /// apply
    pub fn apply(&mut self, param: &O::Param) -> Result<O::Output, Error> {
        self.problem("operator_count", |problem| problem.apply(param))
    }
}

impl<O: CostFunction> Problem<O> {
    /// Compute cost function value
    pub fn cost(&mut self, param: &O::Param) -> Result<O::Output, Error> {
        self.problem("cost_count", |problem| problem.cost(param))
    }
}

impl<O: Gradient> Problem<O> {
    /// Compute gradient
    pub fn gradient(&mut self, param: &O::Param) -> Result<O::Gradient, Error> {
        self.problem("gradient_count", |problem| problem.gradient(param))
    }
}

impl<O: Hessian> Problem<O> {
    /// Compute Hessian
    pub fn hessian(&mut self, param: &O::Param) -> Result<O::Hessian, Error> {
        self.problem("hessian_count", |problem| problem.hessian(param))
    }
}

impl<O: Jacobian> Problem<O> {
    /// Compute Jacobian
    pub fn jacobian(&mut self, param: &O::Param) -> Result<O::Jacobian, Error> {
        self.problem("jacobian_count", |problem| problem.jacobian(param))
    }
}

impl<O: LinearProgram> Problem<O> {
    /// Calls the `c` method of `problem`
    pub fn c(&self) -> Result<Vec<O::Float>, Error> {
        self.problem.as_ref().unwrap().c()
    }

    /// Calls the `b` method of `problem`
    pub fn b(&self) -> Result<Vec<O::Float>, Error> {
        self.problem.as_ref().unwrap().b()
    }

    /// Calls the `A` method of `problem`
    #[allow(non_snake_case)]
    pub fn A(&self) -> Result<Vec<Vec<O::Float>>, Error> {
        self.problem.as_ref().unwrap().A()
    }
}
