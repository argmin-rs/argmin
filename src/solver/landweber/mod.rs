// Copyright 2018 Stefan Kroboth
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
//! [0] Landweber, L. (1951): An iteration formula for Fredholm integral equations of the first
//! kind. Amer. J. Math. 73, 615–624
//! [1] https://en.wikipedia.org/wiki/Landweber_iteration

use crate::prelude::*;
use serde::{Deserialize, Serialize};

/// The Landweber iteration is a solver for ill-posed linear inverse problems.
///
/// In iteration `k`, the new parameter vector `x_{k+1}` is calculated from the previous parameter
/// vector `x_k` and the gradient at `x_k` according to the following update rule:
///
/// `x_{k+1} = x_k - omega * \nabla f(x_k)`
///
/// # Example
///
/// ```rust
/// TODO
/// ```
///
/// # References
///
/// [0] Landweber, L. (1951): An iteration formula for Fredholm integral equations of the first
/// kind. Amer. J. Math. 73, 615–624
/// [1] https://en.wikipedia.org/wiki/Landweber_iteration
#[derive(Serialize, Deserialize)]
pub struct Landweber {
    /// omgea
    omega: f64,
}

impl Landweber {
    /// Constructor
    pub fn new(omega: f64) -> Result<Self, Error> {
        Ok(Landweber { omega })
    }
}

impl<O> Solver<O> for Landweber
where
    O: ArgminOp,
    O::Param: ArgminScaledSub<O::Param, f64, O::Param>,
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
    use crate::send_sync_test;

    send_sync_test!(landweber, Landweber);
}
