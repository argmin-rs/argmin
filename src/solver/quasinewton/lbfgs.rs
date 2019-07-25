// Copyright 2018-2019 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # References:
//!
//! [0] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
//! Springer. ISBN 0-387-30303-0.

use crate::prelude::*;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fmt::Debug;

/// L-BFGS method
///
/// [Example](https://github.com/argmin-rs/argmin/blob/master/examples/lbfgs.rs)
///
/// TODO: Implement compact representation of BFGS updating (Nocedal/Wright p.230)
///
/// # References:
///
/// [0] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(Clone, Serialize, Deserialize)]
pub struct LBFGS<L, P> {
    /// line search
    linesearch: L,
    /// m
    m: usize,
    /// s_{k-1}
    s: VecDeque<P>,
    /// y_{k-1}
    y: VecDeque<P>,
}

impl<L, P> LBFGS<L, P> {
    /// Constructor
    pub fn new(linesearch: L, m: usize) -> Self {
        LBFGS {
            linesearch,
            m,
            s: VecDeque::with_capacity(m),
            y: VecDeque::with_capacity(m),
        }
    }
}

impl<O, L, P> Solver<O> for LBFGS<L, P>
where
    O: ArgminOp<Param = P, Output = f64>,
    O::Param: Clone
        + Serialize
        + DeserializeOwned
        + Debug
        + Default
        + ArgminSub<O::Param, O::Param>
        + ArgminAdd<O::Param, O::Param>
        + ArgminDot<O::Param, f64>
        + ArgminScaledAdd<O::Param, f64, O::Param>
        + ArgminNorm<f64>
        + ArgminMul<f64, O::Param>,
    O::Hessian: Clone + Default + Serialize + DeserializeOwned,
    L: Clone + ArgminLineSearch<O::Param> + Solver<OpWrapper<O>>,
{
    const NAME: &'static str = "L-BFGS";

    fn init(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &IterState<O>,
    ) -> Result<Option<ArgminIterData<O>>, Error> {
        let param = state.get_param();
        let cost = op.apply(&param)?;
        let grad = op.gradient(&param)?;
        Ok(Some(
            ArgminIterData::new().param(param).cost(cost).grad(grad),
        ))
    }

    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &IterState<O>,
    ) -> Result<ArgminIterData<O>, Error> {
        let param = state.get_param();
        let cur_cost = state.get_cost();
        let prev_grad = state.get_grad().unwrap();
        // .unwrap_or_else(|| op.gradient(&param).unwrap());

        let gamma: f64 = if let (Some(ref sk), Some(ref yk)) = (self.s.back(), self.y.back()) {
            sk.dot(*yk) / yk.dot(*yk)
        } else {
            1.0
        };

        // L-BFGS two-loop recursion
        let mut q = prev_grad.clone();
        let cur_m = self.s.len();
        let mut alpha: Vec<f64> = vec![0.0; cur_m];
        let mut rho: Vec<f64> = vec![0.0; cur_m];
        for (i, (ref sk, ref yk)) in self.s.iter().rev().zip(self.y.iter().rev()).enumerate() {
            let sk = *sk;
            let yk = *yk;
            let yksk: f64 = yk.dot(sk);
            let rho_t = 1.0 / yksk;
            let skq: f64 = sk.dot(&q);
            let alpha_t = skq.mul(&rho_t);
            q = q.sub(&yk.mul(&alpha_t));
            rho[cur_m - i - 1] = rho_t;
            alpha[cur_m - i - 1] = alpha_t;
        }
        let mut r = q.mul(&gamma);
        for (i, (ref sk, ref yk)) in self.s.iter().zip(self.y.iter()).enumerate() {
            let sk = *sk;
            let yk = *yk;
            let beta = yk.dot(&r).mul(&rho[i]);
            r = r.add(&sk.mul(&(alpha[i] - beta)));
        }

        self.linesearch.set_search_direction(r.mul(&-1.0));

        // Run solver
        let ArgminResult {
            operator: line_op,
            state:
                IterState {
                    param: xk1,
                    cost: next_cost,
                    ..
                },
        } = Executor::new(
            OpWrapper::new_from_op(&op),
            self.linesearch.clone(),
            param.clone(),
        )
        .grad(prev_grad.clone())
        .cost(cur_cost)
        .ctrlc(false)
        .run()?;

        // take care of function eval counts
        op.consume_op(line_op);

        if state.get_iter() >= self.m as u64 {
            self.s.pop_front();
            self.y.pop_front();
        }

        let grad = op.gradient(&xk1)?;

        self.s.push_back(xk1.sub(&param));
        self.y.push_back(grad.sub(&prev_grad));

        Ok(ArgminIterData::new()
            .param(xk1)
            .cost(next_cost)
            .grad(grad)
            .kv(make_kv!("gamma" => gamma;)))
    }

    fn terminate(&mut self, state: &IterState<O>) -> TerminationReason {
        if state.get_grad().unwrap().norm() < std::f64::EPSILON.sqrt() {
            return TerminationReason::TargetPrecisionReached;
        }
        if (state.get_prev_cost() - state.get_cost()).abs() < std::f64::EPSILON {
            return TerminationReason::NoChangeInCost;
        }
        TerminationReason::NotTerminated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_trait_impl;
    use crate::solver::linesearch::MoreThuenteLineSearch;

    type Operator = MinimalNoOperator;

    test_trait_impl!(lbfgs, LBFGS<Operator, MoreThuenteLineSearch<Operator>>);
}
