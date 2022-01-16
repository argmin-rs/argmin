// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # References:
//!
//! \[0\] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
//! Springer. ISBN 0-387-30303-0.

use crate::core::{
    ArgminFloat, ArgminIterData, ArgminKV, ArgminLineSearch, ArgminOp, ArgminResult,
    DeserializeOwnedAlias, Error, Executor, IterState, OpWrapper, SerializeAlias, Solver,
    TerminationReason,
};
use argmin_math::{ArgminAdd, ArgminDot, ArgminMul, ArgminNorm, ArgminScaledAdd, ArgminSub};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fmt::Debug;

/// L-BFGS method
///
/// TODO: Implement compact representation of BFGS updating (Nocedal/Wright p.230)
///
/// # References:
///
/// \[0\] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct LBFGS<L, P, F> {
    /// line search
    linesearch: L,
    /// m
    m: usize,
    /// s_{k-1}
    s: VecDeque<P>,
    /// y_{k-1}
    y: VecDeque<P>,
    /// Tolerance for the stopping criterion based on the change of the norm on the gradient
    tol_grad: F,
    /// Tolerance for the stopping criterion based on the change of the cost stopping criterion
    tol_cost: F,
}

impl<L, P, F: ArgminFloat> LBFGS<L, P, F> {
    /// Constructor
    pub fn new(linesearch: L, m: usize) -> Self {
        LBFGS {
            linesearch,
            m,
            s: VecDeque::with_capacity(m),
            y: VecDeque::with_capacity(m),
            tol_grad: F::epsilon().sqrt(),
            tol_cost: F::epsilon(),
        }
    }

    /// Sets tolerance for the stopping criterion based on the change of the norm on the gradient
    #[must_use]
    pub fn with_tol_grad(mut self, tol_grad: F) -> Self {
        self.tol_grad = tol_grad;
        self
    }

    /// Sets tolerance for the stopping criterion based on the change of the cost stopping criterion
    #[must_use]
    pub fn with_tol_cost(mut self, tol_cost: F) -> Self {
        self.tol_cost = tol_cost;
        self
    }
}

impl<O, L, P, F> Solver<O> for LBFGS<L, P, F>
where
    O: ArgminOp<Param = P, Output = F, Float = F>,
    O::Param: Clone
        + SerializeAlias
        + DeserializeOwnedAlias
        + Debug
        + Default
        + ArgminSub<O::Param, O::Param>
        + ArgminAdd<O::Param, O::Param>
        + ArgminDot<O::Param, O::Float>
        + ArgminScaledAdd<O::Param, O::Float, O::Param>
        + ArgminNorm<O::Float>
        + ArgminMul<O::Float, O::Param>,
    O::Hessian: Clone + Default + SerializeAlias + DeserializeOwnedAlias,
    L: Clone + ArgminLineSearch<O::Param, O::Float> + Solver<O>,
    // L: Clone + ArgminLineSearch<O::Param, O::Float> + Solver<OpWrapper<O>>,
    F: ArgminFloat,
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

        let gamma: F = if let (Some(sk), Some(yk)) = (self.s.back(), self.y.back()) {
            sk.dot(yk) / yk.dot(yk)
        } else {
            F::from_f64(1.0).unwrap()
        };

        // L-BFGS two-loop recursion
        let mut q = prev_grad.clone();
        let cur_m = self.s.len();
        let mut alpha: Vec<F> = vec![F::from_f64(0.0).unwrap(); cur_m];
        let mut rho: Vec<F> = vec![F::from_f64(0.0).unwrap(); cur_m];
        for (i, (sk, yk)) in self.s.iter().rev().zip(self.y.iter().rev()).enumerate() {
            let yksk: F = yk.dot(sk);
            let rho_t = F::from_f64(1.0).unwrap() / yksk;
            let skq: F = sk.dot(&q);
            let alpha_t = skq.mul(rho_t);
            q = q.sub(&yk.mul(&alpha_t));
            rho[cur_m - i - 1] = rho_t;
            alpha[cur_m - i - 1] = alpha_t;
        }
        let mut r = q.mul(&gamma);
        for (i, (sk, yk)) in self.s.iter().zip(self.y.iter()).enumerate() {
            let beta = yk.dot(&r).mul(rho[i]);
            r = r.add(&sk.mul(&(alpha[i] - beta)));
        }

        self.linesearch
            .set_search_direction(r.mul(&F::from_f64(-1.0).unwrap()));

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
            op.take_op().unwrap(),
            self.linesearch.clone(),
            param.clone(),
        )
        .grad(prev_grad.clone())
        .cost(cur_cost)
        .ctrlc(false)
        .run()?;

        // take back operator and take care of function evaluation counts
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
        if state.get_grad().unwrap().norm() < self.tol_grad {
            return TerminationReason::TargetPrecisionReached;
        }
        if (state.get_prev_cost() - state.get_cost()).abs() < self.tol_cost {
            return TerminationReason::NoChangeInCost;
        }
        TerminationReason::NotTerminated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::MinimalNoOperator;
    use crate::solver::linesearch::MoreThuenteLineSearch;
    use crate::test_trait_impl;

    type Operator = MinimalNoOperator;

    test_trait_impl!(lbfgs, LBFGS<Operator, MoreThuenteLineSearch<Operator, f64>, f64>);

    #[test]
    fn test_tolerances() {
        let linesearch: MoreThuenteLineSearch<f64, f64> =
            MoreThuenteLineSearch::new().c(1e-4, 0.9).unwrap();

        let tol1 = 1e-4;
        let tol2 = 1e-2;

        let LBFGS {
            tol_grad: t1,
            tol_cost: t2,
            ..
        }: LBFGS<_, f64, f64> = LBFGS::new(linesearch, 7)
            .with_tol_grad(tol1)
            .with_tol_cost(tol2);

        assert!((t1 - tol1).abs() < std::f64::EPSILON);
        assert!((t2 - tol2).abs() < std::f64::EPSILON);
    }
}
