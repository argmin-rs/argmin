// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::{
    ArgminOp, CostFunction, Error, Gradient, Hessian, Jacobian, LinearProgram, Modify, Operator,
};
use std::collections::HashMap;
#[cfg(feature = "serde1")]
// use serde::{Deserialize, Serialize};
use std::default::Default;

/// This wraps an operator and keeps track of how often the cost, gradient and Hessian have been
/// computed and how often the modify function has been called. Usually, this is an implementation
/// detail unless a solver is needed within another solver (such as a line search within a gradient
/// descent method).
#[derive(Clone, Debug, Default)]
pub struct OpWrapper<O> {
    /// Operator
    pub op: Option<O>,
    /// Number of cost function evaluations
    pub cost_func_count: u64,
    /// Number of gradient function evaluations
    pub grad_func_count: u64,
    /// Number of Hessian function evaluations
    pub hessian_func_count: u64,
    /// Number of Jacobian function evaluations
    pub jacobian_func_count: u64,
    /// Number of `modify` function evaluations
    pub modify_func_count: u64,
    /// Evaluation counts
    pub counts: HashMap<&'static str, u64>,
}

impl<O: Operator> OpWrapper<O> {
    /// apply
    pub fn apply(&mut self, param: &O::Param) -> Result<O::Output, Error> {
        self.op("operator", |op| op.apply(param))
    }
}

impl<O: CostFunction> OpWrapper<O> {
    /// Compute cost function value
    pub fn cost(&mut self, param: &O::Param) -> Result<O::Output, Error> {
        self.op("cost", |op| op.cost(param))
    }
}

impl<O: Gradient> OpWrapper<O> {
    /// Compute gradient
    pub fn gradient(&mut self, param: &O::Param) -> Result<O::Gradient, Error> {
        self.op("gradient", |op| op.gradient(param))
    }
}

impl<O: Hessian> OpWrapper<O> {
    /// Compute Hessian
    pub fn hessian(&mut self, param: &O::Param) -> Result<O::Hessian, Error> {
        self.op("hessian", |op| op.hessian(param))
    }
}

impl<O: Jacobian> OpWrapper<O> {
    /// Compute Jacobian
    pub fn jacobian(&mut self, param: &O::Param) -> Result<O::Jacobian, Error> {
        self.op("jacobian", |op| op.jacobian(param))
    }
}

impl<O: Modify> OpWrapper<O> {
    /// Compute TODO
    pub fn modify(&mut self, param: &O::Param, extent: O::Float) -> Result<O::Output, Error> {
        self.op("modify", |op| op.modify(param, extent))
    }
}

impl<O> OpWrapper<O> {
    /// general apply
    pub fn op<T, F: FnOnce(&O) -> Result<T, Error>>(
        &mut self,
        name: &'static str,
        func: F,
    ) -> Result<T, Error> {
        let count = self.counts.entry(name).or_insert(0);
        *count += 1;
        func(self.op.as_ref().unwrap())
    }
}

impl<O> OpWrapper<O> {
    /// Construct an `OpWrapper` from an operator
    pub fn new(op: O) -> Self {
        OpWrapper {
            op: Some(op),
            cost_func_count: 0,
            grad_func_count: 0,
            hessian_func_count: 0,
            jacobian_func_count: 0,
            modify_func_count: 0,
            counts: HashMap::new(),
        }
    }
}

impl<O: ArgminOp> OpWrapper<O> {
    /// Calls the `apply` method of `op` and increments `cost_func_count`.
    pub fn apply2(&mut self, param: &O::Param) -> Result<O::Output, Error> {
        self.cost_func_count += 1;
        self.op.as_ref().unwrap().apply2(param)
    }

    /// Calls the `gradient` method of `op` and increments `gradient_func_count`.
    pub fn gradient2(&mut self, param: &O::Param) -> Result<O::Param, Error> {
        self.grad_func_count += 1;
        self.op.as_ref().unwrap().gradient2(param)
    }

    /// Calls the `hessian` method of `op` and increments `hessian_func_count`.
    pub fn hessian2(&mut self, param: &O::Param) -> Result<O::Hessian, Error> {
        self.hessian_func_count += 1;
        self.op.as_ref().unwrap().hessian2(param)
    }

    /// Calls the `jacobian` method of `op` and increments `jacobian_func_count`.
    pub fn jacobian2(&mut self, param: &O::Param) -> Result<O::Jacobian, Error> {
        self.jacobian_func_count += 1;
        self.op.as_ref().unwrap().jacobian2(param)
    }

    /// Calls the `modify` method of `op` and increments `modify_func_count`.
    pub fn modify2(&mut self, param: &O::Param, extent: O::Float) -> Result<O::Param, Error> {
        self.modify_func_count += 1;
        self.op.as_ref().unwrap().modify2(param, extent)
    }

    /// Moves the operator out of the struct and replaces it with `None`
    pub fn take_op(&mut self) -> Option<O> {
        self.op.take()
    }

    /// Consumes an operator by increasing the function call counts of `self` by the ones in
    /// `other`.
    pub fn consume_op(&mut self, other: OpWrapper<O>) {
        self.op = other.op;
        self.cost_func_count += other.cost_func_count;
        self.grad_func_count += other.grad_func_count;
        self.hessian_func_count += other.hessian_func_count;
        self.jacobian_func_count += other.jacobian_func_count;
        self.modify_func_count += other.modify_func_count;
    }

    // pub fn increment_func_counts(
    //     &mut self,
    //     cost: u64,
    //     grad: u64,
    //     hessian: u64,
    //     jacobian: u64,
    //     modify: u64,
    // ) {
    //     self.op = other.op;
    //     self.cost_func_count += cost;
    //     self.grad_func_count += grad;
    //     self.hessian_func_count += hessian;
    //     self.jacobian_func_count += jacobian;
    //     self.modify_func_count += modify;
    // }

    /// Adds function evaluation counts of another operator.
    pub fn consume_func_counts<O2: ArgminOp>(&mut self, other: OpWrapper<O2>) {
        self.cost_func_count += other.cost_func_count;
        self.grad_func_count += other.grad_func_count;
        self.hessian_func_count += other.hessian_func_count;
        self.jacobian_func_count += other.jacobian_func_count;
        self.modify_func_count += other.modify_func_count;
    }

    /// Reset the cost function counts to zero.
    #[must_use]
    pub fn reset(mut self) -> Self {
        self.cost_func_count = 0;
        self.grad_func_count = 0;
        self.hessian_func_count = 0;
        self.jacobian_func_count = 0;
        self.modify_func_count = 0;
        self
    }

    /// Returns the operator `op` by taking ownership of `self`.
    pub fn get_op(self) -> O {
        self.op.unwrap()
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
