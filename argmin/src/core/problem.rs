// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

pub use crate::core::{ArgminError, ArgminFloat, DeserializeOwnedAlias, Error, SerializeAlias};

/// TODO
pub trait Operator {
    /// Type of the parameter vector
    type Param: Clone + SerializeAlias + DeserializeOwnedAlias;
    /// Output of the operator
    type Output: Clone + SerializeAlias + DeserializeOwnedAlias;

    /// Applies the operator to parameters
    fn apply(&self, param: &Self::Param) -> Result<Self::Output, Error>;
}

/// TODO
pub trait CostFunction {
    /// Type of the parameter vector
    type Param: Clone + SerializeAlias + DeserializeOwnedAlias;
    /// Output of the cost function
    type Output: Clone + SerializeAlias + DeserializeOwnedAlias;

    /// Compute cost function
    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error>;
}

/// TODO
pub trait Gradient {
    /// Type of the parameter vector
    type Param: Clone + SerializeAlias + DeserializeOwnedAlias;
    /// Type of the gradient
    type Gradient: Clone + SerializeAlias + DeserializeOwnedAlias;

    /// Compute gradient
    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, Error>;
}

/// TODO
pub trait Hessian {
    /// Type of the parameter vector
    type Param: Clone + SerializeAlias + DeserializeOwnedAlias;
    /// Type of the Hessian
    type Hessian: Clone + SerializeAlias + DeserializeOwnedAlias;

    /// Compute Hessian
    fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, Error>;
}

/// TODO
pub trait Jacobian {
    /// Type of the parameter vector
    type Param: Clone + SerializeAlias + DeserializeOwnedAlias;
    /// Output of the cost function
    type Jacobian: Clone + SerializeAlias + DeserializeOwnedAlias;

    /// Compute Jacobian
    fn jacobian(&self, param: &Self::Param) -> Result<Self::Jacobian, Error>;
}

/// TODO
pub trait Modify {
    /// Type of the parameter vector
    type Param: Clone + SerializeAlias + DeserializeOwnedAlias;
    /// Output TODO
    type Output: Clone + SerializeAlias + DeserializeOwnedAlias;
    /// Precision of floats
    type Float;

    /// Compute Jacobian
    fn modify(&self, param: &Self::Param, _extent: Self::Float) -> Result<Self::Output, Error>;
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
