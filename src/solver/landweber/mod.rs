// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Landweber iteration
//!
//! [Landweber](struct.Landweber.html)
//!
//! # References
//!
//! \[0\] Landweber, L. (1951): An iteration formula for Fredholm integral equations of the first
//! kind. Amer. J. Math. 73, 615–624
//! \[1\] <https://en.wikipedia.org/wiki/Landweber_iteration>

use crate::prelude::*;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// The Landweber iteration is a solver for ill-posed linear inverse problems.
///
/// In iteration `k`, the new parameter vector `x_{k+1}` is calculated from the previous parameter
/// vector `x_k` and the gradient at `x_k` according to the following update rule:
///
/// `x_{k+1} = x_k - omega * \nabla f(x_k)`
///
/// # References
///
/// \[0\] Landweber, L. (1951): An iteration formula for Fredholm integral equations of the first
/// kind. Amer. J. Math. 73, 615–624
/// \[1\] <https://en.wikipedia.org/wiki/Landweber_iteration>
#[derive(Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct Landweber<F> {
    /// omega
    omega: F,
}

impl<F> Landweber<F> {
    /// Constructor
    pub fn new(omega: F) -> Self {
        Landweber { omega }
    }
}

impl<O, F> Solver<O> for Landweber<F>
where
    O: ArgminOp<Float = F>,
    O::Param: ArgminScaledSub<O::Param, O::Float, O::Param>,
    F: ArgminFloat,
{
    const NAME: &'static str = "Landweber";

    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &IterState<O>,
    ) -> Result<ArgminIterData<O>, Error> {
        let param = state.get_param();
        let grad = op.gradient(&param)?;
        let new_param = param.scaled_sub(&self.omega, &grad);
        Ok(ArgminIterData::new().param(new_param))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_trait_impl;

    test_trait_impl!(landweber, Landweber<f64>);
}
