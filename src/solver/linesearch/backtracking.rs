// Copyright 2018-2020 argmin developers
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
/// [Example](https://github.com/argmin-rs/argmin/blob/master/examples/backtracking.rs)
///
/// # References:
///
/// [0] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
///
/// [1] Wikipedia: https://en.wikipedia.org/wiki/Backtracking_line_search
#[derive(Serialize, Deserialize, Clone)]
pub struct BacktrackingLineSearch<P, L, F> {
    /// initial parameter vector
    init_param: P,
    /// initial cost
    init_cost: F,
    /// initial gradient
    init_grad: P,
    /// Search direction
    search_direction: Option<P>,
    /// Contraction factor rho
    rho: F,
    /// Stopping condition
    condition: Box<L>,
    /// alpha
    alpha: F,
}

impl<P: Default, L, F: ArgminFloat> BacktrackingLineSearch<P, L, F> {
    /// Constructor
    pub fn new(condition: L) -> Self {
        BacktrackingLineSearch {
            init_param: P::default(),
            init_cost: F::infinity(),
            init_grad: P::default(),
            search_direction: None,
            rho: F::from_f64(0.9).unwrap(),
            condition: Box::new(condition),
            alpha: F::from_f64(1.0).unwrap(),
        }
    }

    /// Set rho
    pub fn rho(mut self, rho: F) -> Result<Self, Error> {
        if rho <= F::from_f64(0.0).unwrap() || rho >= F::from_f64(1.0).unwrap() {
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

impl<P, L, F> ArgminLineSearch<P, F> for BacktrackingLineSearch<P, L, F>
where
    P: Clone + Serialize + ArgminSub<P, P> + ArgminDot<P, f64> + ArgminScaledAdd<P, f64, P>,
    L: LineSearchCondition<P, F>,
    F: ArgminFloat + Serialize + DeserializeOwned,
{
    /// Set search direction
    fn set_search_direction(&mut self, search_direction: P) {
        self.search_direction = Some(search_direction);
    }

    /// Set initial alpha value
    fn set_init_alpha(&mut self, alpha: F) -> Result<(), Error> {
        if alpha <= F::from_f64(0.0).unwrap() {
            return Err(ArgminError::InvalidParameter {
                text: "LineSearch: Inital alpha must be > 0.".to_string(),
            }
            .into());
        }
        self.alpha = alpha;
        Ok(())
    }
}

impl<P, L, F: ArgminFloat> BacktrackingLineSearch<P, L, F>
where
    P: ArgminScaledAdd<P, F, P>,
    L: LineSearchCondition<P, F>,
{
    fn backtracking_step<O: ArgminOp<Param = P, Output = F, Float = F>>(
        &self,
        op: &mut OpWrapper<O>,
    ) -> Result<ArgminIterData<O>, Error> {
        let new_param = self
            .init_param
            .scaled_add(&self.alpha, self.search_direction.as_ref().unwrap());

        let cur_cost = op.apply(&new_param)?;

        let out = if self.condition.requires_cur_grad() {
            ArgminIterData::new()
                .grad(op.gradient(&new_param)?)
                .param(new_param)
                .cost(cur_cost)
        } else {
            ArgminIterData::new().param(new_param).cost(cur_cost)
        };

        Ok(out)
    }
}

impl<O, P, L, F> Solver<O> for BacktrackingLineSearch<P, L, F>
where
    P: Clone + Default + Serialize + DeserializeOwned + ArgminScaledAdd<P, F, P>,
    O: ArgminOp<Param = P, Output = F, Float = F>,
    L: LineSearchCondition<P, F>,
    F: ArgminFloat,
{
    const NAME: &'static str = "Backtracking Line search";

    fn init(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &IterState<O>,
    ) -> Result<Option<ArgminIterData<O>>, Error> {
        self.init_param = state.get_param();
        let cost = state.get_cost();
        self.init_cost = if cost == F::infinity() {
            op.apply(&self.init_param)?
        } else {
            cost
        };

        self.init_grad = state
            .get_grad()
            .map(Result::Ok)
            .unwrap_or_else(|| op.gradient(&self.init_param))?;

        if self.search_direction.is_none() {
            return Err(ArgminError::NotInitialized {
                text: "BacktrackingLineSearch: search_direction must be set.".to_string(),
            }
            .into());
        }

        let out = self.backtracking_step(op)?;
        Ok(Some(out))
    }

    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        _state: &IterState<O>,
    ) -> Result<ArgminIterData<O>, Error> {
        self.alpha = self.alpha * self.rho;
        self.backtracking_step(op)
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
    use crate::core::MinimalNoOperator;
    use crate::test_trait_impl;

    test_trait_impl!(backtrackinglinesearch,
                    BacktrackingLineSearch<MinimalNoOperator, ArmijoCondition<f64>, f64>);
}
