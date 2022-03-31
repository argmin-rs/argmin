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
    ArgminFloat, CostFunction, DeserializeOwnedAlias, Error, Executor, Gradient, IterState,
    LineSearch, OptimizationResult, Problem, SerializeAlias, Solver, State, TerminationReason, KV,
};
use argmin_math::{ArgminAdd, ArgminDot, ArgminMul, ArgminNorm, ArgminSub};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

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
pub struct LBFGS<L, P, G, F> {
    /// line search
    linesearch: L,
    /// m
    m: usize,
    /// s_{k-1}
    s: VecDeque<P>,
    /// y_{k-1}
    y: VecDeque<G>,
    /// Tolerance for the stopping criterion based on the change of the norm on the gradient
    tol_grad: F,
    /// Tolerance for the stopping criterion based on the change of the cost stopping criterion
    tol_cost: F,
}

impl<L, P, G, F> LBFGS<L, P, G, F>
where
    F: ArgminFloat,
{
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

impl<O, L, P, G, F> Solver<O, IterState<P, G, (), (), F>> for LBFGS<L, P, G, F>
where
    O: CostFunction<Param = P, Output = F> + Gradient<Param = P, Gradient = G>,
    P: Clone
        + SerializeAlias
        + DeserializeOwnedAlias
        + ArgminSub<P, P>
        + ArgminAdd<P, P>
        + ArgminDot<G, F>
        + ArgminMul<F, P>,
    G: Clone
        + SerializeAlias
        + DeserializeOwnedAlias
        + ArgminNorm<F>
        + ArgminSub<G, G>
        + ArgminDot<G, F>
        + ArgminDot<P, F>
        + ArgminMul<F, G>
        + ArgminMul<F, P>,
    L: Clone + LineSearch<P, F> + Solver<O, IterState<P, G, (), (), F>>,
    F: ArgminFloat,
{
    const NAME: &'static str = "L-BFGS";

    fn init(
        &mut self,
        problem: &mut Problem<O>,
        mut state: IterState<P, G, (), (), F>,
    ) -> Result<(IterState<P, G, (), (), F>, Option<KV>), Error> {
        let param = state.take_param().unwrap();
        let cost = problem.cost(&param)?;
        let grad = problem.gradient(&param)?;
        Ok((state.param(param).cost(cost).grad(grad), None))
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        mut state: IterState<P, G, (), (), F>,
    ) -> Result<(IterState<P, G, (), (), F>, Option<KV>), Error> {
        let param = state.take_param().unwrap();
        let cur_cost = state.get_cost();
        let prev_grad = state.take_grad().unwrap();

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
        let mut r: P = q.mul(&gamma);
        for (i, (sk, yk)) in self.s.iter().zip(self.y.iter()).enumerate() {
            let beta: F = yk.dot(&r);
            let beta = beta.mul(rho[i]);
            r = r.add(&sk.mul(&(alpha[i] - beta)));
        }

        self.linesearch
            .set_search_direction(r.mul(&F::from_f64(-1.0).unwrap()));

        // Run solver
        let OptimizationResult {
            problem: line_problem,
            state: mut linesearch_state,
            ..
        } = Executor::new(problem.take_problem().unwrap(), self.linesearch.clone())
            .configure(|config| {
                config
                    .param(param.clone())
                    .grad(prev_grad.clone())
                    .cost(cur_cost)
            })
            .ctrlc(false)
            .run()?;

        let xk1 = linesearch_state.take_param().unwrap();
        let next_cost = linesearch_state.get_cost();

        // take back problem and take care of function evaluation counts
        problem.consume_problem(line_problem);

        if state.get_iter() >= self.m as u64 {
            self.s.pop_front();
            self.y.pop_front();
        }

        let grad = problem.gradient(&xk1)?;

        self.s.push_back(xk1.sub(&param));
        self.y.push_back(grad.sub(&prev_grad));

        Ok((
            state.param(xk1).cost(next_cost).grad(grad),
            Some(make_kv!("gamma" => gamma;)),
        ))
    }

    fn terminate(&mut self, state: &IterState<P, G, (), (), F>) -> TerminationReason {
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
    use crate::solver::linesearch::MoreThuenteLineSearch;
    use crate::test_trait_impl;

    test_trait_impl!(
        lbfgs,
        LBFGS<MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64>, Vec<f64>, Vec<f64>, f64>
    );

    #[test]
    fn test_tolerances() {
        let linesearch: MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64> =
            MoreThuenteLineSearch::new().c(1e-4, 0.9).unwrap();

        let tol1 = 1e-4f64;
        let tol2 = 1e-2;

        let LBFGS {
            tol_grad: t1,
            tol_cost: t2,
            ..
        }: LBFGS<_, Vec<f64>, Vec<f64>, f64> = LBFGS::new(linesearch, 7)
            .with_tol_grad(tol1)
            .with_tol_cost(tol2);

        assert!((t1 - tol1).abs() < std::f64::EPSILON);
        assert!((t2 - tol2).abs() < std::f64::EPSILON);
    }
}
