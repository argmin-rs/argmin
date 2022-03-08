// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::{CostFunction, Error, Gradient, Hessian, Jacobian, Modify, Operator};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// Fake Operators for testing

/// Pseudo operator which is used in tests
#[derive(Clone, Default, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Copy)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct PseudoOperator {}

impl PseudoOperator {
    /// Constructor
    #[allow(dead_code)]
    pub fn new() -> Self {
        PseudoOperator {}
    }
}

impl Operator for PseudoOperator {
    type Param = Vec<f64>;
    type Output = Vec<f64>;

    /// Do nothing.
    fn apply(&self, _p: &Self::Param) -> Result<Self::Output, Error> {
        unimplemented!()
    }
}

impl CostFunction for PseudoOperator {
    type Param = Vec<f64>;
    type Output = f64;

    /// Do nothing.
    fn cost(&self, _p: &Self::Param) -> Result<Self::Output, Error> {
        unimplemented!()
    }
}

impl Gradient for PseudoOperator {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    /// Do nothing.
    fn gradient(&self, _p: &Self::Param) -> Result<Self::Param, Error> {
        unimplemented!()
    }
}

impl Hessian for PseudoOperator {
    type Param = Vec<f64>;
    type Hessian = Vec<Vec<f64>>;

    /// Do nothing.
    fn hessian(&self, _p: &Self::Param) -> Result<Self::Hessian, Error> {
        unimplemented!()
    }
}

impl Jacobian for PseudoOperator {
    type Param = Vec<f64>;
    type Jacobian = Vec<Vec<f64>>;

    /// Do nothing.
    fn jacobian(&self, _p: &Self::Param) -> Result<Self::Jacobian, Error> {
        unimplemented!()
    }
}

impl Modify for PseudoOperator {
    type Param = Vec<f64>;
    type Output = f64;
    type Float = f64;

    /// Do nothing.
    fn modify(&self, _p: &Self::Param, _t: Self::Float) -> Result<Self::Output, Error> {
        unimplemented!()
    }
}
