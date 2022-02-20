// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::{ArgminFloat, ArgminOp, DeserializeOwnedAlias, Error, SerializeAlias};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display};

/// Fake Operators for testing

/// No-op operator with free choice of the types
#[derive(Clone, Default, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Copy)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct NoOperator<T, U, H, J, F> {
    /// Fake parameter
    param: std::marker::PhantomData<T>,
    /// Fake output
    output: std::marker::PhantomData<U>,
    /// Fake Hessian
    hessian: std::marker::PhantomData<H>,
    /// Fake Jacobian
    jacobian: std::marker::PhantomData<J>,
    /// Fake Float
    float: std::marker::PhantomData<F>,
}

impl<T, U, H, J, F> NoOperator<T, U, H, J, F> {
    /// Constructor
    #[allow(dead_code)]
    pub fn new() -> Self {
        NoOperator {
            param: std::marker::PhantomData,
            output: std::marker::PhantomData,
            hessian: std::marker::PhantomData,
            jacobian: std::marker::PhantomData,
            float: std::marker::PhantomData,
        }
    }
}

impl<T, U, H, J, F> Display for NoOperator<T, U, H, J, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "NoOperator")
    }
}

impl<T, U, H, J, F> ArgminOp for NoOperator<T, U, H, J, F>
where
    T: Clone + Default + Debug + Send + Sync + SerializeAlias + DeserializeOwnedAlias,
    U: Clone + Default + Debug + Send + Sync + SerializeAlias + DeserializeOwnedAlias,
    H: Clone + Default + Debug + Send + Sync + SerializeAlias + DeserializeOwnedAlias,
    J: Clone + Default + Debug + Send + Sync + SerializeAlias + DeserializeOwnedAlias,
    F: ArgminFloat,
{
    type Param = T;
    type Output = U;
    type Hessian = H;
    type Jacobian = J;
    type Float = F;

    /// Do nothing, really.
    fn apply2(&self, _p: &Self::Param) -> Result<Self::Output, Error> {
        Ok(Self::Output::default())
    }

    /// Do nothing, really.
    fn gradient2(&self, _p: &Self::Param) -> Result<Self::Param, Error> {
        Ok(Self::Param::default())
    }

    /// Do nothing, really.
    fn hessian2(&self, _p: &Self::Param) -> Result<Self::Hessian, Error> {
        Ok(Self::Hessian::default())
    }

    /// Do nothing, really.
    fn modify2(&self, _p: &Self::Param, _t: Self::Float) -> Result<Self::Param, Error> {
        Ok(Self::Param::default())
    }
}

/// Minimal No-op operator which does nothing, really.
#[derive(Clone, Default, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Copy)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
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
    type Float = f64;

    /// Do nothing, really.
    fn apply2(&self, _p: &Self::Param) -> Result<Self::Output, Error> {
        unimplemented!()
    }

    /// Do nothing, really.
    fn gradient2(&self, _p: &Self::Param) -> Result<Self::Param, Error> {
        unimplemented!()
    }

    /// Do nothing, really.
    fn hessian2(&self, _p: &Self::Param) -> Result<Self::Hessian, Error> {
        unimplemented!()
    }

    /// Do nothing, really.
    fn modify2(&self, _p: &Self::Param, _t: f64) -> Result<Self::Param, Error> {
        unimplemented!()
    }
}
