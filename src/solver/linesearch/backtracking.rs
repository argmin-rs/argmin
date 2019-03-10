// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! * [Backtracking line search](struct.BacktrackingLineSearch.html)

use crate::prelude::*;
use crate::solver::linesearch::condition::*;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

/// The Backtracking line search is a simple method to find a step length which obeys the Armijo
/// (sufficient decrease) condition.
///
/// # Example
///
/// ```rust
/// TODO
/// ```
///
/// # References:
///
/// [0] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
///
/// [1] Wikipedia: https://en.wikipedia.org/wiki/Backtracking_line_search
#[derive(Serialize, Deserialize, Clone)]
pub struct BacktrackingLineSearch<P, L> {
    /// initial parameter vector
    init_param: P,
    /// initial cost
    init_cost: f64,
    /// initial gradient
    init_grad: P,
    /// Search direction
    search_direction: Option<P>,
    /// Contraction factor rho
    rho: f64,
    /// Stopping condition
    condition: Box<L>,
    /// alpha
    alpha: f64,
}

impl<P: Default, L> BacktrackingLineSearch<P, L> {
    /// Constructor
    pub fn new(condition: L) -> Self {
        BacktrackingLineSearch {
            init_param: P::default(),
            init_cost: std::f64::INFINITY,
            init_grad: P::default(),
            search_direction: None,
            rho: 0.9,
            condition: Box::new(condition),
            alpha: 1.0,
        }
    }

    /// Set rho
    pub fn rho(mut self, rho: f64) -> Result<Self, Error> {
        if rho <= 0.0 || rho >= 1.0 {
            return Err(ArgminError::InvalidParameter {
                text: "BacktrackingLineSearch: Contraction factor rho must be in (0, 1)."
                    .to_string(),
            }
            .into());
        }
        self.rho = rho;
        Ok(self)
    }
}

impl<P, L> ArgminLineSearch<P> for BacktrackingLineSearch<P, L>
where
    P: Clone + Serialize + ArgminSub<P, P> + ArgminDot<P, f64> + ArgminScaledAdd<P, f64, P>,
    L: LineSearchCondition<P>,
{
    /// Set search direction
    fn set_search_direction(&mut self, search_direction: P) {
        self.search_direction = Some(search_direction);
    }

    /// Set initial alpha value
    fn set_init_alpha(&mut self, alpha: f64) -> Result<(), Error> {
        if alpha <= 0.0 {
            return Err(ArgminError::InvalidParameter {
                text: "LineSearch: Inital alpha must be > 0.".to_string(),
            }
            .into());
        }
        self.alpha = alpha;
        Ok(())
    }
}

impl<O, P, L> Solver<O> for BacktrackingLineSearch<P, L>
where
    P: Clone
        + Default
        + Serialize
        + DeserializeOwned
        + ArgminSub<P, P>
        + ArgminDot<P, f64>
        + ArgminScaledAdd<P, f64, P>,
    O: ArgminOp<Param = P, Output = f64>,
    L: LineSearchCondition<P>,
{
    const NAME: &'static str = "Backtracking Line search";

    fn init(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &IterState<O>,
    ) -> Result<Option<ArgminIterData<O>>, Error> {
        self.init_param = state.get_param();
        let cost = state.get_cost();
        self.init_cost = if cost == std::f64::INFINITY {
            op.apply(&self.init_param)?
        } else {
            cost
        };

        self.init_grad = state.get_grad().unwrap_or(op.gradient(&self.init_param)?);

        if self.search_direction.is_none() {
            return Err(ArgminError::NotInitialized {
                text: "BacktrackingLineSearch: search_direction must be set.".to_string(),
            }
            .into());
        }

        Ok(None)
    }

    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        _state: &IterState<O>,
    ) -> Result<ArgminIterData<O>, Error> {
        let new_param = self
            .init_param
            .scaled_add(&self.alpha, self.search_direction.as_ref().unwrap());

        let cur_cost = op.apply(&new_param)?;

        self.alpha *= self.rho;

        let mut out = ArgminIterData::new()
            .param(new_param.clone())
            .cost(cur_cost);

        if self.condition.requires_cur_grad() {
            out = out.grad(op.gradient(&new_param)?);
        }

        Ok(out)
    }

    fn terminate(&mut self, state: &IterState<O>) -> TerminationReason {
        if self.condition.eval(
            state.get_cost(),
            state.get_grad().unwrap_or_default(),
            self.init_cost,
            self.init_grad.clone(),
            self.search_direction.clone().unwrap(),
            self.alpha,
        ) {
            TerminationReason::LineSearchConditionMet
        } else {
            TerminationReason::NotTerminated
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::send_sync_test;
    use crate::MinimalNoOperator;

    send_sync_test!(backtrackinglinesearch,
                    BacktrackingLineSearch<MinimalNoOperator, ArmijoCondition>);
}
