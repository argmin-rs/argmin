// Copyright 2018-2019 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

// TODO: Need a way to create new solutions

//! # References:
//!
//! 0. [Wikipedia](https://en.wikipedia.org/wiki/Cuckoo_search)
//! 1. X.-S. Yang; S. Deb (December 2009). Cuckoo search via Lévy flights. World Congress on Nature
//!    & Biologically Inspired Computing (NaBIC 2009). IEEE Publications. pp. 210–214.

use crate::prelude::*;
use rand::prelude::*;
use rand_xorshift::XorShiftRng;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
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
    /// step length for parameter modification
    alpha: f64,
    /// random number generator
    rng: XorShiftRng,
}

/// A nest with eggs
#[derive(Serialize, Deserialize)]
pub struct Nest<O: ArgminOp> {
    eggs: Vec<Egg<O>>,
    cost: O::Output,
}

impl<O: ArgminOp> Nest<O> {
    /// Get best cost in nest
    pub fn get_cost(&self) -> O::Output {
        self.cost.clone()
    }

    /// Replace best egg in nest
    pub fn replace_best(&mut self, egg: Egg<O>) {
        self.cost = egg.get_cost();
        self.eggs[0] = egg;
    }

    /// compute cost
    pub fn compute_cost<F>(&mut self, cost_fun: &mut F) -> Result<(), Error>
    where
        F: FnMut(&O::Param) -> Result<O::Output, Error>,
    {
        for idx in 0..self.eggs.len() {
            self.eggs[idx].compute_cost(cost_fun)?;
        }
        Ok(())
    }
}

/// An egg is round
#[derive(Serialize, Deserialize)]
pub struct Egg<O: ArgminOp> {
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

    /// Get cost
    pub fn get_cost(&self) -> O::Output {
        if let Some(cost) = self.cost.clone() {
            cost
        } else {
            panic!("fix me");
        }
    }

    /// compute cost
    pub fn compute_cost<F>(&mut self, cost_fun: &mut F) -> Result<(), Error>
    where
        F: FnMut(&O::Param) -> Result<O::Output, Error>,
    {
        if self.cost.is_none() {
            self.cost = Some((*cost_fun)(&self.param)?);
        }
        Ok(())
    }
}

impl<O: ArgminOp> PartialEq for Egg<O>
where
    O::Param: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.param == other.param
    }
}

impl<O: ArgminOp> Eq for Egg<O> where O::Param: PartialEq {}

impl<O: ArgminOp> PartialOrd for Egg<O>
where
    O::Param: PartialEq,
    O::Output: Ord,
{
    fn partial_cmp(&self, other: &Egg<O>) -> Option<Ordering> {
        if let (Some(a), Some(b)) = (&self.cost, &other.cost) {
            Some(a.cmp(&b))
        } else {
            None
        }
    }
}

impl<O: ArgminOp> Ord for Egg<O>
where
    O::Param: PartialEq,
    O::Output: Ord,
{
    fn cmp(&self, other: &Egg<O>) -> Ordering {
        if let (Some(a), Some(b)) = (&self.cost, &other.cost) {
            a.cmp(&b)
        } else {
            panic!()
        }
    }
}

impl<O: ArgminOp> CuckooSearch<O> {
    /// Constructor
    pub fn new() -> Self {
        CuckooSearch {
            nests: vec![],
            alpha: 1.0,
            rng: XorShiftRng::from_entropy(),
        }
    }

    /// Add a nest
    pub fn with_nest(mut self, nest: Nest<O>) -> Self {
        self.nests.push(nest);
        self
    }

    /// set alpha
    pub fn alpha(mut self, alpha: f64) -> Result<Self, Error> {
        if alpha <= 0.0 {
            return Err(ArgminError::InvalidParameter {
                text: "CuckooSearch: alpha must be > 0.".to_string(),
            }
            .into());
        }
        self.alpha = alpha;
        Ok(self)
    }

    /// compute cost
    pub fn compute_cost<F>(&mut self, cost_fun: &mut F) -> Result<(), Error>
    where
        F: FnMut(&O::Param) -> Result<O::Output, Error>,
    {
        for idx in 0..self.nests.len() {
            self.nests[idx].compute_cost(cost_fun)?;
        }
        Ok(())
    }
}

impl<O: ArgminOp> Default for CuckooSearch<O> {
    fn default() -> CuckooSearch<O> {
        CuckooSearch::new()
    }
}

impl<O> Solver<O> for CuckooSearch<O>
where
    O: ArgminOp,
    O::Output: PartialOrd,
    O::Param: Default
        + ArgminScaledSub<O::Param, f64, O::Param>
        + ArgminSub<O::Param, O::Param>
        + ArgminMul<f64, O::Param>,
{
    const NAME: &'static str = "Cuckoo search";

    fn init(
        &mut self,
        op: &mut OpWrapper<O>,
        _state: &IterState<O>,
    ) -> Result<Option<ArgminIterData<O>>, Error> {
        self.compute_cost(&mut |x| op.apply(&x))?;
        Ok(None)
    }

    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &IterState<O>,
    ) -> Result<ArgminIterData<O>, Error> {
        let param = state.get_param();
        let param_new = op.modify(&param, 1.0)?;
        let cost_new = op.apply(&param_new)?;
        let nest_idx = self.rng.gen_range(0, self.nests.len());

        if cost_new < self.nests[nest_idx].get_cost() {
            self.nests[nest_idx].replace_best(Egg::lay(param_new).with_cost(cost_new));
        }

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
