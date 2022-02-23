// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::{
    ArgminFloat, CostFunction, DeserializeOwnedAlias, Error, Gradient, Hessian, Jacobian, Modify,
    Operator, SerializeAlias,
};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display};

/// Fake Operators for testing

/// No-op operator with free choice of the types
#[derive(Clone, Default, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Copy)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct NoOperator<P, U, G, H, J, F> {
    /// Fake parameter
    param: std::marker::PhantomData<P>,
    /// Fake output
    output: std::marker::PhantomData<U>,
    /// Fake gradient
    gradient: std::marker::PhantomData<G>,
    /// Fake Hessian
    hessian: std::marker::PhantomData<H>,
    /// Fake Jacobian
    jacobian: std::marker::PhantomData<J>,
    /// Fake Float
    float: std::marker::PhantomData<F>,
}

impl<P, U, G, H, J, F> NoOperator<P, U, G, H, J, F> {
    /// Constructor
    #[allow(dead_code)]
    pub fn new() -> Self {
        NoOperator {
            param: std::marker::PhantomData,
            output: std::marker::PhantomData,
            gradient: std::marker::PhantomData,
            hessian: std::marker::PhantomData,
            jacobian: std::marker::PhantomData,
            float: std::marker::PhantomData,
        }
    }
}

impl<P, U, G, H, J, F> Display for NoOperator<P, U, G, H, J, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "NoOperator")
    }
}

impl<P, U, G, H, J, F> Operator for NoOperator<P, U, G, H, J, F>
where
    P: Clone + Default + Debug + Send + Sync + SerializeAlias + DeserializeOwnedAlias,
    U: Clone + Default + Debug + Send + Sync + SerializeAlias + DeserializeOwnedAlias,
    G: Clone + Default + Debug + Send + Sync + SerializeAlias + DeserializeOwnedAlias,
    H: Clone + Default + Debug + Send + Sync + SerializeAlias + DeserializeOwnedAlias,
    J: Clone + Default + Debug + Send + Sync + SerializeAlias + DeserializeOwnedAlias,
    F: ArgminFloat,
{
    type Param = P;
    type Output = U;

    /// Do nothing, really.
    fn apply(&self, _p: &Self::Param) -> Result<Self::Output, Error> {
        Ok(Self::Output::default())
    }
}

impl<P, U, G, H, J, F> CostFunction for NoOperator<P, U, G, H, J, F>
where
    P: Clone + Default + Debug + Send + Sync + SerializeAlias + DeserializeOwnedAlias,
    U: Clone + Default + Debug + Send + Sync + SerializeAlias + DeserializeOwnedAlias,
    G: Clone + Default + Debug + Send + Sync + SerializeAlias + DeserializeOwnedAlias,
    H: Clone + Default + Debug + Send + Sync + SerializeAlias + DeserializeOwnedAlias,
    J: Clone + Default + Debug + Send + Sync + SerializeAlias + DeserializeOwnedAlias,
    F: ArgminFloat,
{
    type Param = P;
    type Output = U;

    /// Do nothing, really.
    fn cost(&self, _p: &Self::Param) -> Result<Self::Output, Error> {
        Ok(Self::Output::default())
    }
}

impl<P, U, G, H, J, F> Gradient for NoOperator<P, U, G, H, J, F>
where
    P: Clone + Default + Debug + Send + Sync + SerializeAlias + DeserializeOwnedAlias,
    U: Clone + Default + Debug + Send + Sync + SerializeAlias + DeserializeOwnedAlias,
    G: Clone + Default + Debug + Send + Sync + SerializeAlias + DeserializeOwnedAlias,
    H: Clone + Default + Debug + Send + Sync + SerializeAlias + DeserializeOwnedAlias,
    J: Clone + Default + Debug + Send + Sync + SerializeAlias + DeserializeOwnedAlias,
    F: ArgminFloat,
{
    type Param = P;
    type Gradient = G;

    /// Do nothing, really.
    fn gradient(&self, _p: &Self::Param) -> Result<Self::Gradient, Error> {
        Ok(Self::Gradient::default())
    }
}

impl<P, U, G, H, J, F> Hessian for NoOperator<P, U, G, H, J, F>
where
    P: Clone + Default + Debug + Send + Sync + SerializeAlias + DeserializeOwnedAlias,
    U: Clone + Default + Debug + Send + Sync + SerializeAlias + DeserializeOwnedAlias,
    G: Clone + Default + Debug + Send + Sync + SerializeAlias + DeserializeOwnedAlias,
    H: Clone + Default + Debug + Send + Sync + SerializeAlias + DeserializeOwnedAlias,
    J: Clone + Default + Debug + Send + Sync + SerializeAlias + DeserializeOwnedAlias,
    F: ArgminFloat,
{
    type Param = P;
    type Hessian = H;

    /// Do nothing, really.
    fn hessian(&self, _p: &Self::Param) -> Result<Self::Hessian, Error> {
        Ok(Self::Hessian::default())
    }
}

impl<P, U, G, H, J, F> Jacobian for NoOperator<P, U, G, H, J, F>
where
    P: Clone + Default + Debug + Send + Sync + SerializeAlias + DeserializeOwnedAlias,
    U: Clone + Default + Debug + Send + Sync + SerializeAlias + DeserializeOwnedAlias,
    G: Clone + Default + Debug + Send + Sync + SerializeAlias + DeserializeOwnedAlias,
    H: Clone + Default + Debug + Send + Sync + SerializeAlias + DeserializeOwnedAlias,
    J: Clone + Default + Debug + Send + Sync + SerializeAlias + DeserializeOwnedAlias,
    F: ArgminFloat,
{
    type Param = P;
    type Jacobian = J;

    /// Do nothing, really.
    fn jacobian(&self, _p: &Self::Param) -> Result<Self::Jacobian, Error> {
        Ok(Self::Jacobian::default())
    }
}

impl<P, U, G, H, J, F> Modify for NoOperator<P, U, G, H, J, F>
where
    P: Clone + Default + Debug + Send + Sync + SerializeAlias + DeserializeOwnedAlias,
    U: Clone + Default + Debug + Send + Sync + SerializeAlias + DeserializeOwnedAlias,
    G: Clone + Default + Debug + Send + Sync + SerializeAlias + DeserializeOwnedAlias,
    H: Clone + Default + Debug + Send + Sync + SerializeAlias + DeserializeOwnedAlias,
    J: Clone + Default + Debug + Send + Sync + SerializeAlias + DeserializeOwnedAlias,
    F: ArgminFloat,
{
    type Param = P;
    type Output = U;
    type Float = F;

    /// Do nothing, really.
    fn modify(&self, _p: &Self::Param, _t: Self::Float) -> Result<Self::Output, Error> {
        Ok(Self::Output::default())
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

impl Operator for MinimalNoOperator {
    type Param = Vec<f64>;
    type Output = Vec<f64>;

    /// Do nothing, really.
    fn apply(&self, _p: &Self::Param) -> Result<Self::Output, Error> {
        unimplemented!()
    }
}

impl CostFunction for MinimalNoOperator {
    type Param = Vec<f64>;
    type Output = f64;

    /// Do nothing, really.
    fn cost(&self, _p: &Self::Param) -> Result<Self::Output, Error> {
        unimplemented!()
    }
}

impl Gradient for MinimalNoOperator {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    /// Do nothing, really.
    fn gradient(&self, _p: &Self::Param) -> Result<Self::Param, Error> {
        unimplemented!()
    }
}

impl Hessian for MinimalNoOperator {
    type Param = Vec<f64>;
    type Hessian = Vec<Vec<f64>>;

    /// Do nothing, really.
    fn hessian(&self, _p: &Self::Param) -> Result<Self::Hessian, Error> {
        unimplemented!()
    }
}

impl Jacobian for MinimalNoOperator {
    type Param = Vec<f64>;
    type Jacobian = Vec<Vec<f64>>;

    /// Do nothing, really.
    fn jacobian(&self, _p: &Self::Param) -> Result<Self::Jacobian, Error> {
        unimplemented!()
    }
}

impl Modify for MinimalNoOperator {
    type Param = Vec<f64>;
    type Output = f64;
    type Float = f64;

    /// Do nothing, really.
    fn modify(&self, _p: &Self::Param, _t: Self::Float) -> Result<Self::Output, Error> {
        unimplemented!()
    }
}
