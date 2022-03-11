// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::{CostFunction, Error, Gradient, Hessian, Jacobian, LinearProgram, Operator};
use std::collections::HashMap;
#[cfg(feature = "serde1")]
use std::default::Default;

/// This wraps an operator and keeps track of how often the cost, gradient and Hessian have been
/// computed. Usually, this is an implementation detail unless a solver is needed within another
/// solver (such as a line search within a gradient descent method).
#[derive(Clone, Debug, Default)]
pub struct OpWrapper<O> {
    /// Operator
    pub op: Option<O>,
    /// Evaluation counts
    pub counts: HashMap<&'static str, u64>,
}

impl<O> OpWrapper<O> {
    /// Gives access to the stored operator `op` via the closure `func` and keeps track of how many
    /// times the function has been called. The function counts will be passed to observers labelled
    /// with `counts_string`. Per convention, `counts_string` is chosen as `<something>_count`.
    ///
    /// Example
    /// ```
    /// # use argmin::core::{PseudoOperator, OpWrapper, CostFunction};
    /// # let mut op_wrapper = OpWrapper::new(PseudoOperator::new());
    /// # let param = vec![1.0f64, 0.0];
    /// let cost = op_wrapper.op("cost_count", |op| op.cost(&param));
    /// # assert_eq!(op_wrapper.counts["cost_count"], 1)
    /// ```
    ///
    /// This is typically used when designing a trait which optimization problems need to implement
    /// for certain solvers. For instance, for a trait `Anneal` used in Simulated Annealing, one
    /// would write the following to enable the solver to call `.anneal(...)` on an `OpWrapper`
    /// directly:
    ///
    /// ```
    /// # // needs to reimplement OpWrapper because doctests run in dedicated crate where it is not
    /// # // possible to `impl` a type from another crate.
    /// # // Probably somewhat related to https://github.com/rust-lang/rust/issues/50784
    /// # use argmin::core::{Error};
    /// # use std::collections::HashMap;
    /// #
    /// # pub struct OpWrapper<O> {
    /// #     pub op: Option<O>,
    /// #     pub counts: HashMap<&'static str, u64>,
    /// # }
    /// # impl<O> OpWrapper<O> {
    /// #     pub fn op<T, F: FnOnce(&O) -> Result<T, Error>>(
    /// #         &mut self,
    /// #         counts_string: &'static str,
    /// #         func: F,
    /// #     ) -> Result<T, Error> {
    /// #         let count = self.counts.entry(counts_string).or_insert(0);
    /// #         *count += 1;
    /// #         func(self.op.as_ref().unwrap())
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
    /// impl<O: Anneal> OpWrapper<O> {
    ///     pub fn anneal(&mut self, param: &O::Param, extent: O::Float) -> Result<O::Output, Error> {
    ///         self.op("anneal_count", |op| op.anneal(param, extent))
    ///     }
    /// }
    ///
    /// // ...
    ///
    /// # struct Problem {}
    /// # impl Anneal for Problem {
    /// #     type Param = ();
    /// #     type Output = ();
    /// #     type Float = f64;
    /// #
    /// #     fn anneal(&self, param: &Self::Param, _extent: Self::Float) -> Result<Self::Output, Error> {
    /// #         Ok(())
    /// #     }
    /// # }
    /// # let mut op_wrapper = OpWrapper { op: Some(Problem {}), counts: HashMap::new() };
    /// # let param = ();
    /// let new_param = op_wrapper.anneal(&param, 1.0f64);
    /// # assert_eq!(op_wrapper.counts["anneal_count"], 1)
    /// ```
    ///
    /// Note that this will unfortunately only work inside the `argmin` crate itself due to the fact
    /// that it is not possible to `impl` a type from another crate. Therefore if one implements a
    /// solver outside of argmin, `.op(...)` has to be called directly as shown in the first
    /// example.
    pub fn op<T, F: FnOnce(&O) -> Result<T, Error>>(
        &mut self,
        counts_string: &'static str,
        func: F,
    ) -> Result<T, Error> {
        let count = self.counts.entry(counts_string).or_insert(0);
        *count += 1;
        func(self.op.as_ref().unwrap())
    }
}

impl<O> OpWrapper<O> {
    /// Construct an `OpWrapper` from an operator
    pub fn new(op: O) -> Self {
        OpWrapper {
            op: Some(op),
            counts: HashMap::new(),
        }
    }
}

impl<O> OpWrapper<O> {
    /// Moves the operator out of the struct and replaces it with `None`
    pub fn take_op(&mut self) -> Option<O> {
        self.op.take()
    }

    /// Consumes an operator by increasing the function call counts of `self` by the ones in
    /// `other`.
    pub fn consume_op(&mut self, mut other: OpWrapper<O>) {
        self.op = Some(other.take_op().unwrap());
        self.consume_func_counts(other);
    }

    /// Adds function evaluation counts of another operator.
    pub fn consume_func_counts<O2>(&mut self, other: OpWrapper<O2>) {
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

    /// Returns the operator `op` by taking ownership of `self`.
    pub fn get_op(self) -> O {
        self.op.unwrap()
    }
}
impl<O: Operator> OpWrapper<O> {
    /// apply
    pub fn apply(&mut self, param: &O::Param) -> Result<O::Output, Error> {
        self.op("operator_count", |op| op.apply(param))
    }
}

impl<O: CostFunction> OpWrapper<O> {
    /// Compute cost function value
    pub fn cost(&mut self, param: &O::Param) -> Result<O::Output, Error> {
        self.op("cost_count", |op| op.cost(param))
    }
}

impl<O: Gradient> OpWrapper<O> {
    /// Compute gradient
    pub fn gradient(&mut self, param: &O::Param) -> Result<O::Gradient, Error> {
        self.op("gradient_count", |op| op.gradient(param))
    }
}

impl<O: Hessian> OpWrapper<O> {
    /// Compute Hessian
    pub fn hessian(&mut self, param: &O::Param) -> Result<O::Hessian, Error> {
        self.op("hessian_count", |op| op.hessian(param))
    }
}

impl<O: Jacobian> OpWrapper<O> {
    /// Compute Jacobian
    pub fn jacobian(&mut self, param: &O::Param) -> Result<O::Jacobian, Error> {
        self.op("jacobian_count", |op| op.jacobian(param))
    }
}

impl<O: LinearProgram> OpWrapper<O> {
    /// Calls the `c` method of `op`
    pub fn c(&self) -> Result<Vec<O::Float>, Error> {
        self.op.as_ref().unwrap().c()
    }

    /// Calls the `b` method of `op`
    pub fn b(&self) -> Result<Vec<O::Float>, Error> {
        self.op.as_ref().unwrap().b()
    }

    /// Calls the `A` method of `op`
    #[allow(non_snake_case)]
    pub fn A(&self) -> Result<Vec<Vec<O::Float>>, Error> {
        self.op.as_ref().unwrap().A()
    }
}
