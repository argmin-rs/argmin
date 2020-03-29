// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::{ArgminOp, Error};
use serde::{Deserialize, Serialize};

/// This wraps an operator and keeps track of how often the cost, gradient and Hessian have been
/// computed and how often the modify function has been called. Usually, this is an implementation
/// detail unless a solver is needed within another solver (such as a line search within a gradient
/// descent method), then it may be necessary to wrap the operator in an OpWrapper.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct OpWrapper<O: ArgminOp> {
    /// Operator
    op: O,
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
}

impl<O: ArgminOp> OpWrapper<O> {
    /// Constructor
    pub fn new(op: &O) -> Self {
        OpWrapper {
            op: op.clone(),
            cost_func_count: 0,
            grad_func_count: 0,
            hessian_func_count: 0,
            jacobian_func_count: 0,
            modify_func_count: 0,
        }
    }

    /// Constructor (moves op)
    pub fn new_move(op: O) -> Self {
        OpWrapper {
            op,
            cost_func_count: 0,
            grad_func_count: 0,
            hessian_func_count: 0,
            jacobian_func_count: 0,
            modify_func_count: 0,
        }
    }

    /// Calls the `apply` method of `op` and increments `cost_func_count`.
    pub fn apply(&mut self, param: &O::Param) -> Result<O::Output, Error> {
        self.cost_func_count += 1;
        self.op.apply(param)
    }

    /// Calls the `gradient` method of `op` and increments `gradient_func_count`.
    pub fn gradient(&mut self, param: &O::Param) -> Result<O::Param, Error> {
        self.grad_func_count += 1;
        self.op.gradient(param)
    }

    /// Calls the `hessian` method of `op` and increments `hessian_func_count`.
    pub fn hessian(&mut self, param: &O::Param) -> Result<O::Hessian, Error> {
        self.hessian_func_count += 1;
        self.op.hessian(param)
    }

    /// Calls the `jacobian` method of `op` and increments `jacobian_func_count`.
    pub fn jacobian(&mut self, param: &O::Param) -> Result<O::Jacobian, Error> {
        self.jacobian_func_count += 1;
        self.op.jacobian(param)
    }

    /// Calls the `modify` method of `op` and increments `modify_func_count`.
    pub fn modify(&mut self, param: &O::Param, extent: f64) -> Result<O::Param, Error> {
        self.modify_func_count += 1;
        self.op.modify(param, extent)
    }

    /// Consumes an operator by increasing the function call counts of `self` by the ones in
    /// `other`.
    pub fn consume_op<O2: ArgminOp>(&mut self, other: OpWrapper<O2>) {
        self.cost_func_count += other.cost_func_count;
        self.grad_func_count += other.grad_func_count;
        self.hessian_func_count += other.hessian_func_count;
        self.jacobian_func_count += other.jacobian_func_count;
        self.modify_func_count += other.modify_func_count;
    }

    /// Reset the cost function counts to zero
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
        self.op
    }

    /// Returns a clone of the operator `op`.
    pub fn clone_op(&self) -> O {
        self.op.clone()
    }

    /// Creates a new `OpWrapper<O>` from another `OpWrapper<O>` by cloning the `op` and
    /// initializing all counts with `0`.
    pub fn new_from_op(op: &OpWrapper<O>) -> Self {
        Self::new(&op.clone_op())
    }
}

/// The OpWrapper<O> should behave just like any other `ArgminOp`
impl<O: ArgminOp> ArgminOp for OpWrapper<O> {
    type Param = O::Param;
    type Output = O::Output;
    type Hessian = O::Hessian;
    type Jacobian = O::Jacobian;

    fn apply(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        self.op.apply(param)
    }

    fn gradient(&self, param: &Self::Param) -> Result<Self::Param, Error> {
        self.op.gradient(param)
    }

    fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, Error> {
        self.op.hessian(param)
    }

    fn jacobian(&self, param: &Self::Param) -> Result<Self::Jacobian, Error> {
        self.op.jacobian(param)
    }

    fn modify(&self, param: &Self::Param, extent: f64) -> Result<Self::Param, Error> {
        self.op.modify(param, extent)
    }
}
