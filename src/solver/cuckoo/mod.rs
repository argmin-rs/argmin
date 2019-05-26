// Copyright 2018-2019 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # References:
//!
//! 0. [Wikipedia](https://en.wikipedia.org/wiki/Cuckoo_search)
//! 1. X.-S. Yang; S. Deb (December 2009). Cuckoo search via Lévy flights. World Congress on Nature
//!    & Biologically Inspired Computing (NaBIC 2009). IEEE Publications. pp. 210–214.

use crate::prelude::*;
use serde::{Deserialize, Serialize};
use std::default::Default;

/// Cuckoo search
///
/// [Example](https://github.com/argmin-rs/argmin/blob/master/examples/cuckoo.rs)
///
/// # References:
///
/// 0. [Wikipedia](https://en.wikipedia.org/wiki/Cuckoo_search)
/// 1. X.-S. Yang; S. Deb (December 2009). Cuckoo search via Lévy flights. World Congress on Nature
///    & Biologically Inspired Computing (NaBIC 2009). IEEE Publications. pp. 210–214.
#[derive(Serialize, Deserialize)]
pub struct CuckooSearch<O: ArgminOp> {
    /// Nests
    nests: Vec<Nest<O>>,
    // /// gamma
    // gamma: f64,
}

/// A nest with eggs
#[derive(Serialize, Deserialize)]
struct Nest<O: ArgminOp> {
    eggs: Vec<Egg<O>>,
    cost: O::Output,
}

/// An egg is round
#[derive(Serialize, Deserialize)]
struct Egg<O: ArgminOp> {
    param: O::Param,
    cost: Option<O::Output>,
}

impl<O: ArgminOp> Egg<O> {
    /// Lay an egg
    pub fn lay(param: O::Param) -> Self {
        Egg { param, cost: None }
    }

    /// Add cost to egg
    pub fn with_cost(mut self, cost: O::Output) -> Self {
        self.cost = Some(cost);
        self
    }

    /// compute cost
    pub fn compute_cost<F>(&mut self, mut cost_fun: F) -> Result<(), Error>
    where
        F: FnMut(&O::Param) -> Result<O::Output, Error>,
    {
        if self.cost.is_none() {
            self.cost = Some((cost_fun)(&self.param)?);
        }
        Ok(())
    }
}

impl<O: ArgminOp> CuckooSearch<O> {
    /// Constructor
    pub fn new() -> Self {
        CuckooSearch {
            nests: vec![],
            // gamma: 1.0,
        }
    }

    // /// set gamma
    // pub fn gamma(mut self, gamma: f64) -> Result<Self, Error> {
    //     if gamma <= 0.0 || gamma > 1.0 {
    //         return Err(ArgminError::InvalidParameter {
    //             text: "CuckooSearch: gamma must be in  (0, 1].".to_string(),
    //         }
    //         .into());
    //     }
    //     self.gamma = gamma;
    //     Ok(self)
    // }
}

impl<O: ArgminOp> Default for CuckooSearch<O> {
    fn default() -> CuckooSearch<O> {
        CuckooSearch::new()
    }
}

impl<O> Solver<O> for CuckooSearch<O>
where
    O: ArgminOp,
    O::Param: Default
        + ArgminScaledSub<O::Param, f64, O::Param>
        + ArgminSub<O::Param, O::Param>
        + ArgminMul<f64, O::Param>,
{
    const NAME: &'static str = "Cuckoo search";

    fn next_iter(
        &mut self,
        _op: &mut OpWrapper<O>,
        _state: &IterState<O>,
    ) -> Result<ArgminIterData<O>, Error> {
        unimplemented!()

        // Ok(ArgminIterData::new()
        //     .param(new_param)
        //     .cost(residuals.norm()))
    }

    fn terminate(&mut self, _state: &IterState<O>) -> TerminationReason {
        // if (state.get_prev_cost() - state.get_cost()).abs() < std::f64::EPSILON.sqrt() {
        //     return TerminationReason::NoChangeInCost;
        // }
        TerminationReason::NotTerminated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::send_sync_test;

    send_sync_test!(cuckoo_search, CuckooSearch<MinimalNoOperator>);
}
