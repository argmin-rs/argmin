// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # `ArgminResult`
//!
//! Return type of the solvers. Includes the final parameter vector, the final cost, the number of
//! iterations, whether it terminated and the reason of termination.

use crate::core::{ArgminOp, IterState};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// This is returned by the `Executor` after the solver is run on the operator.
///
/// TODO: Think about removing this, as returning the IterState may be much better
#[derive(Clone, Serialize, Deserialize)]
pub struct ArgminResult<O: ArgminOp> {
    /// operator
    pub operator: O,
    /// iteration state
    pub state: IterState<O>,
}

impl<O: ArgminOp> ArgminResult<O> {
    /// Constructor
    pub fn new(operator: O, state: IterState<O>) -> Self {
        ArgminResult { operator, state }
    }
}

impl<O> std::fmt::Display for ArgminResult<O>
where
    O: ArgminOp,
    O::Param: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(f, "ArgminResult:")?;
        writeln!(f, "    param (best):  {:?}", self.state.get_best_param())?;
        writeln!(f, "    cost (best):   {}", self.state.get_best_cost())?;
        writeln!(f, "    iters (best):  {}", self.state.get_last_best_iter())?;
        writeln!(f, "    iters (best):  {}", self.state.get_last_best_iter())?;
        writeln!(f, "    iters (total): {}", self.state.get_iter())?;
        writeln!(
            f,
            "    termination: {}",
            self.state.get_termination_reason()
        )?;
        writeln!(f, "    time:        {:?}", self.state.get_time())?;
        Ok(())
    }
}

impl<O: ArgminOp> PartialEq for ArgminResult<O> {
    fn eq(&self, other: &ArgminResult<O>) -> bool {
        (self.state.get_cost() - other.state.get_cost()).abs() < std::f64::EPSILON
    }
}

impl<O: ArgminOp> Eq for ArgminResult<O> {}

impl<O: ArgminOp> Ord for ArgminResult<O> {
    fn cmp(&self, other: &ArgminResult<O>) -> Ordering {
        let t = self.state.get_cost() - other.state.get_cost();
        if t.abs() < std::f64::EPSILON {
            Ordering::Equal
        } else if t > 0.0 {
            Ordering::Greater
        } else {
            Ordering::Less
        }
    }
}

impl<O: ArgminOp> PartialOrd for ArgminResult<O> {
    fn partial_cmp(&self, other: &ArgminResult<O>) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::MinimalNoOperator;

    send_sync_test!(argmin_result, ArgminResult<MinimalNoOperator>);
}
