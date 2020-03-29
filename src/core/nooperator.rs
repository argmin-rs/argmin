// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::{ArgminOp, Error};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display};

/// Fake Operators for testing

/// No-op operator with free choice of the types
#[derive(
    Clone, Default, Debug, Serialize, Deserialize, Eq, PartialEq, Ord, PartialOrd, Hash, Copy,
)]
pub struct NoOperator<T, U, H, J> {
    /// Fake parameter
    param: std::marker::PhantomData<T>,
    /// Fake output
    output: std::marker::PhantomData<U>,
    /// Fake Hessian
    hessian: std::marker::PhantomData<H>,
    /// Fake Jacobian
    jacobian: std::marker::PhantomData<J>,
}

impl<T, U, H, J> NoOperator<T, U, H, J> {
    /// Constructor
    #[allow(dead_code)]
    pub fn new() -> Self {
        NoOperator {
            param: std::marker::PhantomData,
            output: std::marker::PhantomData,
            hessian: std::marker::PhantomData,
            jacobian: std::marker::PhantomData,
        }
    }
}

impl<T, U, H, J> Display for NoOperator<T, U, H, J> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "NoOperator")
    }
}

impl<T, U, H, J> ArgminOp for NoOperator<T, U, H, J>
where
    T: Clone + Default + Debug + Send + Sync + Serialize + DeserializeOwned,
    U: Clone + Default + Debug + Send + Sync + Serialize + DeserializeOwned,
    H: Clone + Default + Debug + Send + Sync + Serialize + DeserializeOwned,
    J: Clone + Default + Debug + Send + Sync + Serialize + DeserializeOwned,
{
    type Param = T;
    type Output = U;
    type Hessian = H;
    type Jacobian = J;

    /// Do nothing, really.
    fn apply(&self, _p: &Self::Param) -> Result<Self::Output, Error> {
        Ok(Self::Output::default())
    }

    /// Do nothing, really.
    fn gradient(&self, _p: &Self::Param) -> Result<Self::Param, Error> {
        Ok(Self::Param::default())
    }

    /// Do nothing, really.
    fn hessian(&self, _p: &Self::Param) -> Result<Self::Hessian, Error> {
        Ok(Self::Hessian::default())
    }

    /// Do nothing, really.
    fn modify(&self, _p: &Self::Param, _t: f64) -> Result<Self::Param, Error> {
        Ok(Self::Param::default())
    }
}

/// Minimal No-op operator which does nothing, really.
#[derive(
    Clone, Default, Debug, Serialize, Deserialize, Eq, PartialEq, Ord, PartialOrd, Hash, Copy,
)]
pub struct MinimalNoOperator {}

/// No-op operator with fixed types (See `ArgminOp` impl on `MinimalNoOperator`)
impl MinimalNoOperator {
    /// Constructor
    #[allow(dead_code)]
    pub fn new() -> Self {
        MinimalNoOperator {}
    }
}

impl Display for MinimalNoOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "MinimalNoOperator")
    }
}

impl ArgminOp for MinimalNoOperator {
    type Param = Vec<f64>;
    type Output = f64;
    type Hessian = Vec<Vec<f64>>;
    type Jacobian = Vec<f64>;

    /// Do nothing, really.
    fn apply(&self, _p: &Self::Param) -> Result<Self::Output, Error> {
        unimplemented!()
    }

    /// Do nothing, really.
    fn gradient(&self, _p: &Self::Param) -> Result<Self::Param, Error> {
        unimplemented!()
    }

    /// Do nothing, really.
    fn hessian(&self, _p: &Self::Param) -> Result<Self::Hessian, Error> {
        unimplemented!()
    }

    /// Do nothing, really.
    fn modify(&self, _p: &Self::Param, _t: f64) -> Result<Self::Param, Error> {
        unimplemented!()
    }
}
