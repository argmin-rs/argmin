// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::{ArgminFloat, Error, SendAlias, SyncAlias};
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use std::collections::HashMap;

/// Wrapper around problems defined by users.
///
/// Keeps track of how many times methods such as `apply`, `cost`, `gradient`, `jacobian`,
/// `hessian`, `anneal` and so on are called. It is used to pass the problem from one iteration of
/// a solver to the next.
#[derive(Clone, Debug, Default)]
pub struct Problem<O> {
    /// Problem defined by user
    pub problem: Option<O>,
    /// Keeps track of how often methods of `problem` have been called.
    pub counts: HashMap<&'static str, u64>,
}

impl<O> Problem<O> {
    /// Wraps a problem into an instance of `Problem`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::Problem;
    /// # use std::collections::HashMap;
    /// #
    /// # #[derive(Eq, PartialEq, Debug)]
    /// # struct UserDefinedProblem {};
    /// #
    /// let wrapped_problem = Problem::new(UserDefinedProblem {});
    /// #
    /// # assert_eq!(wrapped_problem.problem.unwrap(), UserDefinedProblem {});
    /// # assert_eq!(wrapped_problem.counts, HashMap::new());
    /// ```
    pub fn new(problem: O) -> Self {
        Problem {
            problem: Some(problem),
            counts: HashMap::new(),
        }
    }

    /// Gives access to the stored `problem` via the closure `func` and keeps track of how many
    /// times the function has been called. The function counts will be passed to observers labeled
    /// with `counts_string`. Per convention, `counts_string` is chosen as `<something>_count`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{test_utils::TestProblem, Problem, CostFunction};
    /// # let mut problem = Problem::new(TestProblem::new());
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
    /// # use argmin::core::Error;
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

    /// Gives access to the stored `problem` via the closure `func` and keeps track of how many
    /// times the function has been called. In contrast to the `problem` method, this also allows
    /// to pass the number of parameter vectors which will be processed by the underlying problem.
    /// This is used by the `bulk_*` methods, which process multiple parameters at once.
    /// The function counts will be passed to observers labeled with `counts_string`.
    /// Per convention, `counts_string` is chosen as `<something>_count`.
    pub fn bulk_problem<T, F: FnOnce(&O) -> Result<T, Error>>(
        &mut self,
        counts_string: &'static str,
        num_param_vecs: usize,
        func: F,
    ) -> Result<T, Error> {
        let count = self.counts.entry(counts_string).or_insert(0);
        *count += num_param_vecs as u64;
        func(self.problem.as_ref().unwrap())
    }

    /// Returns the internally stored problem and replaces it with `None`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::Problem;
    /// # use std::collections::HashMap;
    /// #
    /// # #[derive(Eq, PartialEq, Debug)]
    /// # struct UserDefinedProblem {};
    /// #
    /// let mut problem = Problem::new(UserDefinedProblem {});
    /// let user_problem: Option<UserDefinedProblem> = problem.take_problem();
    ///
    /// assert_eq!(user_problem.unwrap(), UserDefinedProblem {});
    /// assert!(problem.problem.is_none());
    /// # assert_eq!(problem.counts, HashMap::new());
    /// ```
    pub fn take_problem(&mut self) -> Option<O> {
        self.problem.take()
    }

    /// Consumes another instance of `Problem`. The internally stored user defined problem of the
    /// passed `Problem` instance is moved to `Self`. The function evaluation counts are
    /// merged/summed up.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::Problem;
    /// #
    /// # #[derive(Eq, PartialEq, Debug, Clone)]
    /// # struct UserDefinedProblem {};
    /// #
    /// let mut problem1 = Problem::new(UserDefinedProblem {});
    ///
    /// // Simulate function evaluation counts in `problem1`
    /// problem1.counts.insert("cost_count", 2);
    ///
    /// // Take the internally stored problem such that `None` remains in its place.
    /// let _ = problem1.take_problem();
    /// assert!(problem1.problem.is_none());
    ///
    /// let mut problem2 = Problem::new(UserDefinedProblem {});
    ///
    /// // Simulate function evaluation counts in `problem2`
    /// problem2.counts.insert("cost_count", 1);
    /// problem2.counts.insert("gradient_count", 4);
    ///
    /// // `problem1` consumes `problem2` by moving its internally stored user defined problem and
    /// // by merging the function evaluation counts
    /// problem1.consume_problem(problem2);
    ///
    /// // `problem1` now holds a problem of type `UserDefinedProblem` (taken from `problem2`)
    /// assert_eq!(problem1.problem.unwrap(), UserDefinedProblem {});
    ///
    /// // The function evaluation counts have been merged
    /// assert_eq!(problem1.counts["cost_count"], 3);
    /// assert_eq!(problem1.counts["gradient_count"], 4);
    /// ```
    pub fn consume_problem(&mut self, mut other: Problem<O>) {
        self.problem = Some(other.take_problem().unwrap());
        self.consume_func_counts(other);
    }

    /// Consumes another instance of `Problem` by summing ob the function evaluation counts.
    /// In contrast to `consume_problem`, the internally stored `problem` remains untouched.
    /// Therefore the two internally stored problems do not need to be of the same type.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::Problem;
    /// #
    /// # #[derive(Eq, PartialEq, Debug, Clone)]
    /// # struct UserDefinedProblem1 {};
    /// #
    /// # #[derive(Eq, PartialEq, Debug, Clone)]
    /// # struct UserDefinedProblem2 {};
    /// #
    /// let mut problem1 = Problem::new(UserDefinedProblem1 {});
    ///
    /// // Simulate function evaluation counts in `problem1`.
    /// problem1.counts.insert("cost_count", 2);
    ///
    /// // Take the internally stored problem such that `None` remains in its place.
    /// let _ = problem1.take_problem();
    /// assert!(problem1.problem.is_none());
    ///
    /// let mut problem2 = Problem::new(UserDefinedProblem2 {});
    ///
    /// // Simulate function evaluation counts in `problem2`
    /// problem2.counts.insert("cost_count", 1);
    /// problem2.counts.insert("gradient_count", 4);
    ///
    /// // `problem1` consumes `problem2` by merging the function evaluation counts.
    /// problem1.consume_func_counts(problem2);
    ///
    /// // The internally stored problem remains being `None` (in contrast to `consume_problem`).
    /// assert!(problem1.problem.is_none());
    ///
    /// // The function evaluation counts have been merged.
    /// assert_eq!(problem1.counts["cost_count"], 3);
    /// assert_eq!(problem1.counts["gradient_count"], 4);
    /// ```
    pub fn consume_func_counts<O2>(&mut self, other: Problem<O2>) {
        for (k, v) in other.counts.iter() {
            let count = self.counts.entry(k).or_insert(0);
            *count += v
        }
    }

    /// Resets the function evaluation counts to zero.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::Problem;
    /// #
    /// # #[derive(Eq, PartialEq, Debug, Clone)]
    /// # struct UserDefinedProblem {};
    /// #
    /// let mut problem = Problem::new(UserDefinedProblem {});
    ///
    /// // Simulate function evaluation counts in `problem1`.
    /// problem.counts.insert("cost_count", 2);
    /// problem.counts.insert("gradient_count", 4);
    ///
    /// assert_eq!(problem.counts["cost_count"], 2);
    /// assert_eq!(problem.counts["gradient_count"], 4);
    ///
    /// // Set function evaluation counts to 0
    /// problem.reset();
    ///
    /// assert_eq!(problem.counts["cost_count"], 0);
    /// assert_eq!(problem.counts["gradient_count"], 0);
    /// ```
    pub fn reset(&mut self) {
        for (_, v) in self.counts.iter_mut() {
            *v = 0;
        }
    }

    /// Returns the internally stored user defined problem by consuming `Self`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::Problem;
    /// #
    /// # #[derive(Eq, PartialEq, Debug, Clone)]
    /// # struct UserDefinedProblem {};
    /// #
    /// let problem = Problem::new(UserDefinedProblem {});
    ///
    /// let user_problem = problem.get_problem();
    ///
    /// assert_eq!(user_problem.unwrap(), UserDefinedProblem {});
    /// ```
    pub fn get_problem(self) -> Option<O> {
        self.problem
    }
}

/// Defines the application of an operator to a parameter vector.
///
/// # Example
///
/// ```
/// use argmin::core::{Operator, Error};
/// use argmin_math::ArgminDot;
///
/// struct Model {
///     matrix: Vec<Vec<f64>>,
/// }
///
/// impl Operator for Model {
///     type Param = Vec<f64>;
///     type Output = Vec<f64>;
///
///     /// Multiply matrix `self.matrix` with vector `param`
///     fn apply(&self, param: &Self::Param) -> Result<Self::Output, Error> {
///         Ok(self.matrix.dot(param))
///     }
/// }
/// ```
pub trait Operator {
    /// Type of the parameter vector
    type Param;
    /// Type of the return value of the operator
    type Output;

    /// Applies the operator to parameters
    fn apply(&self, param: &Self::Param) -> Result<Self::Output, Error>;

    bulk!(apply, Self::Param, Self::Output);
}

/// Defines computation of a cost function value
///
/// # Example
///
/// ```
/// use argmin::core::{CostFunction, Error};
/// use argmin_testfunctions::rosenbrock;
///
/// struct Rosenbrock {}
///
/// impl CostFunction for Rosenbrock {
///     type Param = Vec<f64>;
///     type Output = f64;
///
///     /// Compute Rosenbrock function
///     fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
///         Ok(rosenbrock(param))
///     }
/// }
/// ```
pub trait CostFunction {
    /// Type of the parameter vector
    type Param;
    /// Type of the return value of the cost function
    type Output;

    /// Compute cost function
    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error>;

    bulk!(cost, Self::Param, Self::Output);
}

/// Defines the computation of the gradient.
///
/// # Example
///
/// ```
/// use argmin::core::{Gradient, Error};
/// # fn compute_gradient(_a: &[f64]) -> Vec<f64> { vec![] }
///
/// struct Rosenbrock {}
///
/// impl Gradient for Rosenbrock {
///     type Param = Vec<f64>;
///     type Gradient = Vec<f64>;
///
///     /// Compute gradient of rosenbrock function
///     fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, Error> {
///         Ok(compute_gradient(param))
///     }
/// }
/// ```
pub trait Gradient {
    /// Type of the parameter vector
    type Param;
    /// Type of the gradient
    type Gradient;

    /// Compute gradient
    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, Error>;

    bulk!(gradient, Self::Param, Self::Gradient);
}

/// Defines the computation of the Hessian.
///
/// # Example
///
/// ```
/// use argmin::core::{Hessian, Error};
/// # fn compute_hessian(_a: &[f64]) -> Vec<f64> { vec![] }
///
/// struct Rosenbrock {}
///
/// impl Hessian for Rosenbrock {
///     type Param = Vec<f64>;
///     type Hessian = Vec<f64>;
///
///     /// Compute gradient of rosenbrock function
///     fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, Error> {
///         Ok(compute_hessian(&param))
///     }
/// }
pub trait Hessian {
    /// Type of the parameter vector
    type Param;
    /// Type of the Hessian
    type Hessian;

    /// Compute Hessian
    fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, Error>;

    bulk!(hessian, Self::Param, Self::Hessian);
}

/// Defines the computation of the Jacobian.
///
/// # Example
///
/// ```
/// use argmin::core::{Jacobian, Error};
///
/// struct Problem {}
///
/// # fn problem_jacobian(p: &[f64]) -> Vec<Vec<f64>> {
/// #     vec![vec![1.0f64, 2.0f64], vec![1.0f64, 2.0f64]]
/// # }
/// #
/// impl Jacobian for Problem {
///     type Param = Vec<f64>;
///     type Jacobian = Vec<Vec<f64>>;
///
///     fn jacobian(&self, p: &Self::Param) -> Result<Self::Jacobian, Error> {
///         Ok(problem_jacobian(p))
///     }
/// }
/// ```
pub trait Jacobian {
    /// Type of the parameter vector
    type Param;
    /// Type of the Jacobian
    type Jacobian;

    /// Compute Jacobian
    fn jacobian(&self, param: &Self::Param) -> Result<Self::Jacobian, Error>;

    bulk!(jacobian, Self::Param, Self::Jacobian);
}

/// Defines a linear Program
///
/// # Example
///
/// ```
/// use argmin::core::{LinearProgram, Error};
///
/// struct Problem {}
///
/// impl LinearProgram for Problem {
///     type Param = Vec<f64>;
///     type Float = f64;
///
///     fn c(&self) -> Result<Vec<Self::Float>, Error> {
///         Ok(vec![1.0, 2.0])
///     }
///
///     fn b(&self) -> Result<Vec<Self::Float>, Error> {
///         Ok(vec![3.0, 4.0])
///     }
///
///     fn A(&self) -> Result<Vec<Vec<Self::Float>>, Error> {
///         Ok(vec![vec![5.0, 6.0], vec![7.0, 8.0]])
///     }
/// }
/// ```
pub trait LinearProgram {
    /// Type of the parameter vector
    type Param;
    /// Precision of floats
    type Float: ArgminFloat;

    /// TODO c for linear programs
    /// Those three could maybe be merged into a single function; name unclear
    fn c(&self) -> Result<Vec<Self::Float>, Error> {
        Err(argmin_error!(
            NotImplemented,
            "Method `c` of LinearProgram trait not implemented!"
        ))
    }

    /// TODO b for linear programs
    fn b(&self) -> Result<Vec<Self::Float>, Error> {
        Err(argmin_error!(
            NotImplemented,
            "Method `b` of LinearProgram trait not implemented!"
        ))
    }

    /// TODO A for linear programs
    #[allow(non_snake_case)]
    fn A(&self) -> Result<Vec<Vec<Self::Float>>, Error> {
        Err(argmin_error!(
            NotImplemented,
            "Method `A` of LinearProgram trait not implemented!"
        ))
    }
}

/// Wraps a call to `apply` defined in the `Operator` trait and as such allows to call `apply` on
/// an instance of `Problem`. Internally, the number of evaluations of `apply` is counted.
impl<O: Operator> Problem<O> {
    /// Calls `apply` defined in the `Operator` trait and keeps track of the number of evaluations.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{Problem, Operator, Error};
    /// #
    /// # #[derive(Eq, PartialEq, Debug, Clone)]
    /// # struct UserDefinedProblem {};
    /// #
    /// # impl Operator for UserDefinedProblem {
    /// #     type Param = Vec<f64>;
    /// #     type Output = Vec<f64>;
    /// #
    /// #     fn apply(&self, param: &Self::Param) -> Result<Self::Output, Error> {
    /// #         Ok(vec![1.0f64, 1.0f64])
    /// #     }
    /// # }
    /// // `UserDefinedProblem` implements `Operator`.
    /// let mut problem1 = Problem::new(UserDefinedProblem {});
    ///
    /// let param = vec![2.0f64, 1.0f64];
    ///
    /// let res = problem1.apply(&param);
    ///
    /// assert_eq!(problem1.counts["operator_count"], 1);
    /// # assert_eq!(res.unwrap(), vec![1.0f64, 1.0f64]);
    /// ```
    pub fn apply(&mut self, param: &O::Param) -> Result<O::Output, Error> {
        self.problem("operator_count", |problem| problem.apply(param))
    }

    /// Calls `bulk_apply` defined in the `Operator` trait and keeps track of the number of
    /// evaluations.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{Problem, Operator, Error};
    /// #
    /// # #[derive(Eq, PartialEq, Debug, Clone)]
    /// # struct UserDefinedProblem {};
    /// #
    /// # impl Operator for UserDefinedProblem {
    /// #     type Param = Vec<f64>;
    /// #     type Output = Vec<f64>;
    /// #
    /// #     fn apply(&self, param: &Self::Param) -> Result<Self::Output, Error> {
    /// #         Ok(vec![1.0f64, 1.0f64])
    /// #     }
    /// # }
    /// // `UserDefinedProblem` implements `Operator`.
    /// let mut problem1 = Problem::new(UserDefinedProblem {});
    ///
    /// let param1 = vec![2.0f64, 1.0f64];
    /// let param2 = vec![3.0f64, 5.0f64];
    /// let params = vec![&param1, &param2];
    ///
    /// let res = problem1.bulk_apply(&params);
    ///
    /// assert_eq!(problem1.counts["operator_count"], 2);
    /// # let res = res.unwrap();
    /// # assert_eq!(res[0], vec![1.0f64, 1.0f64]);
    /// # assert_eq!(res[1], vec![1.0f64, 1.0f64]);
    /// ```
    pub fn bulk_apply<P>(&mut self, params: &[P]) -> Result<Vec<O::Output>, Error>
    where
        P: std::borrow::Borrow<O::Param> + SyncAlias,
        O::Output: SendAlias,
        O: SyncAlias,
    {
        self.bulk_problem("operator_count", params.len(), |problem| {
            problem.bulk_apply(params)
        })
    }
}

/// Wraps a call to `cost` defined in the `CostFunction` trait and as such allows to call `cost` on
/// an instance of `Problem`. Internally, the number of evaluations of `cost` is counted.
impl<O: CostFunction> Problem<O> {
    /// Calls `cost` defined in the `CostFunction` trait and keeps track of the number of
    /// evaluations.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{Problem, CostFunction, Error};
    /// #
    /// # #[derive(Eq, PartialEq, Debug, Clone)]
    /// # struct UserDefinedProblem {};
    /// #
    /// # impl CostFunction for UserDefinedProblem {
    /// #     type Param = Vec<f64>;
    /// #     type Output = f64;
    /// #
    /// #     fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
    /// #         Ok(4.0f64)
    /// #     }
    /// # }
    /// // `UserDefinedProblem` implements `CostFunction`.
    /// let mut problem1 = Problem::new(UserDefinedProblem {});
    ///
    /// let param = vec![2.0f64, 1.0f64];
    ///
    /// let res = problem1.cost(&param);
    ///
    /// assert_eq!(problem1.counts["cost_count"], 1);
    /// # assert_eq!(res.unwrap(), 4.0f64);
    /// ```
    pub fn cost(&mut self, param: &O::Param) -> Result<O::Output, Error> {
        self.problem("cost_count", |problem| problem.cost(param))
    }

    /// Calls `bulk_cost` defined in the `CostFunction` trait and keeps track of the number of
    /// evaluations.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{Problem, CostFunction, Error};
    /// #
    /// # #[derive(Eq, PartialEq, Debug, Clone)]
    /// # struct UserDefinedProblem {};
    /// #
    /// # impl CostFunction for UserDefinedProblem {
    /// #     type Param = Vec<f64>;
    /// #     type Output = f64;
    /// #
    /// #     fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
    /// #         Ok(4.0f64)
    /// #     }
    /// # }
    /// // `UserDefinedProblem` implements `CostFunction`.
    /// let mut problem1 = Problem::new(UserDefinedProblem {});
    ///
    /// let param1 = vec![2.0f64, 1.0f64];
    /// let param2 = vec![3.0f64, 5.0f64];
    /// let params = vec![&param1, &param2];
    ///
    /// let res = problem1.bulk_cost(&params);
    ///
    /// assert_eq!(problem1.counts["cost_count"], 2);
    /// # let res = res.unwrap();
    /// # assert_eq!(res[0], 4.0f64);
    /// # assert_eq!(res[1], 4.0f64);
    /// ```
    pub fn bulk_cost<P>(&mut self, params: &[P]) -> Result<Vec<O::Output>, Error>
    where
        P: std::borrow::Borrow<O::Param> + SyncAlias,
        O::Output: SendAlias,
        O: SyncAlias,
    {
        self.bulk_problem("cost_count", params.len(), |problem| {
            problem.bulk_cost(params)
        })
    }
}

/// Wraps a call to `gradient` defined in the `Gradient` trait and as such allows to call `gradient` on
/// an instance of `Problem`. Internally, the number of evaluations of `gradient` is counted.
impl<O: Gradient> Problem<O> {
    /// Calls `gradient` defined in the `Gradient` trait and keeps track of the number of
    /// evaluations.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{Problem, Gradient, Error};
    /// #
    /// # #[derive(Eq, PartialEq, Debug, Clone)]
    /// # struct UserDefinedProblem {};
    /// #
    /// # impl Gradient for UserDefinedProblem {
    /// #     type Param = Vec<f64>;
    /// #     type Gradient = Vec<f64>;
    /// #
    /// #     fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, Error> {
    /// #         Ok(vec![1.0f64, 1.0f64])
    /// #     }
    /// # }
    /// // `UserDefinedProblem` implements `Gradient`.
    /// let mut problem1 = Problem::new(UserDefinedProblem {});
    ///
    /// let param = vec![2.0f64, 1.0f64];
    ///
    /// let res = problem1.gradient(&param);
    ///
    /// assert_eq!(problem1.counts["gradient_count"], 1);
    /// # assert_eq!(res.unwrap(), vec![1.0f64, 1.0f64]);
    /// ```
    pub fn gradient(&mut self, param: &O::Param) -> Result<O::Gradient, Error> {
        self.problem("gradient_count", |problem| problem.gradient(param))
    }

    /// Calls `bulk_gradient` defined in the `Gradient` trait and keeps track of the number of
    /// evaluations.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{Problem, Gradient, Error};
    /// #
    /// # #[derive(Eq, PartialEq, Debug, Clone)]
    /// # struct UserDefinedProblem {};
    /// #
    /// # impl Gradient for UserDefinedProblem {
    /// #     type Param = Vec<f64>;
    /// #     type Gradient = Vec<f64>;
    /// #
    /// #     fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, Error> {
    /// #         Ok(vec![1.0f64, 1.0f64])
    /// #     }
    /// # }
    /// // `UserDefinedProblem` implements `Gradient`.
    /// let mut problem1 = Problem::new(UserDefinedProblem {});
    ///
    /// let param1 = vec![2.0f64, 1.0f64];
    /// let param2 = vec![3.0f64, 5.0f64];
    /// let params = vec![&param1, &param2];
    ///
    /// let res = problem1.bulk_gradient(&params);
    ///
    /// assert_eq!(problem1.counts["gradient_count"], 2);
    /// # let res = res.unwrap();
    /// # assert_eq!(res[0], vec![1.0f64, 1.0f64]);
    /// # assert_eq!(res[1], vec![1.0f64, 1.0f64]);
    /// ```
    pub fn bulk_gradient<P>(&mut self, params: &[P]) -> Result<Vec<O::Gradient>, Error>
    where
        P: std::borrow::Borrow<O::Param> + SyncAlias,
        O::Gradient: SendAlias,
        O: SyncAlias,
    {
        self.bulk_problem("gradient_count", params.len(), |problem| {
            problem.bulk_gradient(params)
        })
    }
}

/// Wraps a call to `hessian` defined in the `Hessian` trait and as such allows to call `hessian` on
/// an instance of `Problem`. Internally, the number of evaluations of `hessian` is counted.
impl<O: Hessian> Problem<O> {
    /// Calls `hessian` defined in the `Hessian` trait and keeps track of the number of evaluations.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{Problem, Hessian, Error};
    /// #
    /// # #[derive(Eq, PartialEq, Debug, Clone)]
    /// # struct UserDefinedProblem {};
    /// #
    /// # impl Hessian for UserDefinedProblem {
    /// #     type Param = Vec<f64>;
    /// #     type Hessian = Vec<Vec<f64>>;
    /// #
    /// #     fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, Error> {
    /// #         Ok(vec![vec![1.0f64, 0.0f64], vec![0.0f64, 1.0f64]])
    /// #     }
    /// # }
    /// // `UserDefinedProblem` implements `Hessian`.
    /// let mut problem1 = Problem::new(UserDefinedProblem {});
    ///
    /// let param = vec![2.0f64, 1.0f64];
    ///
    /// let res = problem1.hessian(&param);
    ///
    /// assert_eq!(problem1.counts["hessian_count"], 1);
    /// # assert_eq!(res.unwrap(), vec![vec![1.0f64, 0.0f64], vec![0.0f64, 1.0f64]]);
    /// ```
    pub fn hessian(&mut self, param: &O::Param) -> Result<O::Hessian, Error> {
        self.problem("hessian_count", |problem| problem.hessian(param))
    }

    /// Calls `bulk_hessian` defined in the `Hessian` trait and keeps track of the number of
    /// evaluations.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{Problem, Hessian, Error};
    /// #
    /// # #[derive(Eq, PartialEq, Debug, Clone)]
    /// # struct UserDefinedProblem {};
    /// #
    /// # impl Hessian for UserDefinedProblem {
    /// #     type Param = Vec<f64>;
    /// #     type Hessian = Vec<Vec<f64>>;
    /// #
    /// #     fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, Error> {
    /// #         Ok(vec![vec![1.0f64, 0.0f64], vec![0.0f64, 1.0f64]])
    /// #     }
    /// # }
    /// // `UserDefinedProblem` implements `Hessian`.
    /// let mut problem1 = Problem::new(UserDefinedProblem {});
    ///
    /// let param1 = vec![2.0f64, 1.0f64];
    /// let param2 = vec![3.0f64, 5.0f64];
    /// let params = vec![&param1, &param2];
    ///
    /// let res = problem1.bulk_hessian(&params);
    ///
    /// assert_eq!(problem1.counts["hessian_count"], 2);
    /// # let res = res.unwrap();
    /// # assert_eq!(res[0], vec![vec![1.0f64, 0.0f64], vec![0.0f64, 1.0f64]]);
    /// # assert_eq!(res[1], vec![vec![1.0f64, 0.0f64], vec![0.0f64, 1.0f64]]);
    /// ```
    pub fn bulk_hessian<P>(&mut self, params: &[P]) -> Result<Vec<O::Hessian>, Error>
    where
        P: std::borrow::Borrow<O::Param> + SyncAlias,
        O::Hessian: SendAlias,
        O: SyncAlias,
    {
        self.bulk_problem("hessian_count", params.len(), |problem| {
            problem.bulk_hessian(params)
        })
    }
}

/// Wraps a call to `jacobian` defined in the `Jacobian` trait and as such allows to call `jacobian`
/// on an instance of `Problem`. Internally, the number of evaluations of `jacobian` is counted.
impl<O: Jacobian> Problem<O> {
    /// Calls `jacobian` defined in the `Jacobian` trait and keeps track of the number of
    /// evaluations.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{Problem, Jacobian, Error};
    /// #
    /// # #[derive(Eq, PartialEq, Debug, Clone)]
    /// # struct UserDefinedProblem {};
    /// #
    /// # impl Jacobian for UserDefinedProblem {
    /// #     type Param = Vec<f64>;
    /// #     type Jacobian = Vec<Vec<f64>>;
    /// #
    /// #     fn jacobian(&self, param: &Self::Param) -> Result<Self::Jacobian, Error> {
    /// #         Ok(vec![vec![1.0f64, 0.0f64], vec![0.0f64, 1.0f64]])
    /// #     }
    /// # }
    /// // `UserDefinedProblem` implements `Jacobian`.
    /// let mut problem1 = Problem::new(UserDefinedProblem {});
    ///
    /// let param = vec![2.0f64, 1.0f64];
    ///
    /// let res = problem1.jacobian(&param);
    ///
    /// assert_eq!(problem1.counts["jacobian_count"], 1);
    /// # assert_eq!(res.unwrap(), vec![vec![1.0f64, 0.0f64], vec![0.0f64, 1.0f64]]);
    /// ```
    pub fn jacobian(&mut self, param: &O::Param) -> Result<O::Jacobian, Error> {
        self.problem("jacobian_count", |problem| problem.jacobian(param))
    }

    /// Calls `bulk_jacobian` defined in the `Jacobian` trait and keeps track of the number of
    /// evaluations.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{Problem, Jacobian, Error};
    /// #
    /// # #[derive(Eq, PartialEq, Debug, Clone)]
    /// # struct UserDefinedProblem {};
    /// #
    /// # impl Jacobian for UserDefinedProblem {
    /// #     type Param = Vec<f64>;
    /// #     type Jacobian = Vec<Vec<f64>>;
    /// #
    /// #     fn jacobian(&self, param: &Self::Param) -> Result<Self::Jacobian, Error> {
    /// #         Ok(vec![vec![1.0f64, 0.0f64], vec![0.0f64, 1.0f64]])
    /// #     }
    /// # }
    /// // `UserDefinedProblem` implements `Jacobian`.
    /// let mut problem1 = Problem::new(UserDefinedProblem {});
    ///
    /// let params = vec![vec![2.0f64, 1.0f64], vec![3.0f64, 5.0f64]];
    ///
    /// let res = problem1.bulk_jacobian(&params);
    ///
    /// assert_eq!(problem1.counts["jacobian_count"], 2);
    /// # let res = res.unwrap();
    /// # assert_eq!(res[0], vec![vec![1.0f64, 0.0f64], vec![0.0f64, 1.0f64]]);
    /// # assert_eq!(res[1], vec![vec![1.0f64, 0.0f64], vec![0.0f64, 1.0f64]]);
    /// ```
    pub fn bulk_jacobian<P>(&mut self, params: &[P]) -> Result<Vec<O::Jacobian>, Error>
    where
        P: std::borrow::Borrow<O::Param> + SyncAlias,
        O::Jacobian: SendAlias,
        O: SyncAlias,
    {
        self.bulk_problem("jacobian_count", params.len(), |problem| {
            problem.bulk_jacobian(params)
        })
    }
}

/// Wraps a calls to `c`, `b` and `A` defined in the `LinearProgram` trait and as such allows to
/// call those methods on an instance of `Problem`.
impl<O: LinearProgram> Problem<O> {
    /// Calls `c` defined in the `LinearProgram` trait.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{Problem, LinearProgram, Error};
    /// #
    /// # #[derive(Eq, PartialEq, Debug, Clone)]
    /// # struct UserDefinedProblem {};
    /// #
    /// # impl LinearProgram for UserDefinedProblem {
    /// #     type Param = Vec<f64>;
    /// #     type Float = f64;
    /// #
    /// #     fn c(&self) -> Result<Vec<Self::Float>, Error> {
    /// #         Ok(vec![4.0f64, 3.0f64])
    /// #     }
    /// #
    /// #     fn b(&self) -> Result<Vec<Self::Float>, Error> {
    /// #         Ok(vec![3.0f64, 2.0f64])
    /// #     }
    /// #
    /// #     fn A(&self) -> Result<Vec<Vec<Self::Float>>, Error> {
    /// #         Ok(vec![vec![1.0f64, 2.0f64], vec![3.0f64, 2.0f64]])
    /// #     }
    /// # }
    /// // `UserDefinedProblem` implements `LinearProgram`.
    /// let mut problem1 = Problem::new(UserDefinedProblem {});
    ///
    /// let c = problem1.c();
    /// let b = problem1.b();
    ///
    /// # assert_eq!(c.unwrap(), vec![4.0f64, 3.0f64]);
    /// ```
    pub fn c(&self) -> Result<Vec<O::Float>, Error> {
        self.problem.as_ref().unwrap().c()
    }

    /// Calls `b` defined in the `LinearProgram` trait.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{Problem, LinearProgram, Error};
    /// #
    /// # #[derive(Eq, PartialEq, Debug, Clone)]
    /// # struct UserDefinedProblem {};
    /// #
    /// # impl LinearProgram for UserDefinedProblem {
    /// #     type Param = Vec<f64>;
    /// #     type Float = f64;
    /// #
    /// #     fn c(&self) -> Result<Vec<Self::Float>, Error> {
    /// #         Ok(vec![4.0f64, 3.0f64])
    /// #     }
    /// #
    /// #     fn b(&self) -> Result<Vec<Self::Float>, Error> {
    /// #         Ok(vec![3.0f64, 2.0f64])
    /// #     }
    /// #
    /// #     fn A(&self) -> Result<Vec<Vec<Self::Float>>, Error> {
    /// #         Ok(vec![vec![1.0f64, 2.0f64], vec![3.0f64, 2.0f64]])
    /// #     }
    /// # }
    /// // `UserDefinedProblem` implements `LinearProgram`.
    /// let mut problem1 = Problem::new(UserDefinedProblem {});
    ///
    /// let b = problem1.b();
    ///
    /// # assert_eq!(b.unwrap(), vec![3.0f64, 2.0f64]);
    /// ```
    pub fn b(&self) -> Result<Vec<O::Float>, Error> {
        self.problem.as_ref().unwrap().b()
    }

    /// Calls `A` defined in the `LinearProgram` trait.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::core::{Problem, LinearProgram, Error};
    /// #
    /// # #[derive(Eq, PartialEq, Debug, Clone)]
    /// # struct UserDefinedProblem {};
    /// #
    /// # impl LinearProgram for UserDefinedProblem {
    /// #     type Param = Vec<f64>;
    /// #     type Float = f64;
    /// #
    /// #     fn c(&self) -> Result<Vec<Self::Float>, Error> {
    /// #         Ok(vec![4.0f64, 3.0f64])
    /// #     }
    /// #
    /// #     fn b(&self) -> Result<Vec<Self::Float>, Error> {
    /// #         Ok(vec![3.0f64, 2.0f64])
    /// #     }
    /// #
    /// #     fn A(&self) -> Result<Vec<Vec<Self::Float>>, Error> {
    /// #         Ok(vec![vec![1.0f64, 2.0f64], vec![3.0f64, 2.0f64]])
    /// #     }
    /// # }
    /// // `UserDefinedProblem` implements `LinearProgram`.
    /// let mut problem1 = Problem::new(UserDefinedProblem {});
    ///
    /// let a = problem1.A();
    ///
    /// # assert_eq!(a.unwrap(), vec![vec![1.0f64, 2.0f64], vec![3.0f64, 2.0f64]]);
    /// ```
    #[allow(non_snake_case)]
    pub fn A(&self) -> Result<Vec<Vec<O::Float>>, Error> {
        self.problem.as_ref().unwrap().A()
    }
}
