// Copyright 2019-2022 argmin developers
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
    ArgminError, ArgminFloat, ArgminKV, CostFunction, DeserializeOwnedAlias, Error, Executor,
    Gradient, IterState, LineSearch, OpWrapper, OptimizationResult, SerializeAlias, Solver,
    TerminationReason,
};
use argmin_math::{ArgminAdd, ArgminDot, ArgminMul, ArgminNorm, ArgminSub};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// SR1 method (broken!)
///
/// # References:
///
/// \[0\] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct SR1<L, H, F> {
    /// parameter for skipping rule
    r: F,
    /// Init inverse Hessian
    init_inv_hessian: Option<H>,
    /// line search
    linesearch: L,
    /// Tolerance for the stopping criterion based on the change of the norm on the gradient
    tol_grad: F,
    /// Tolerance for the stopping criterion based on the change of the cost stopping criterion
    tol_cost: F,
}

impl<L, H, F> SR1<L, H, F>
where
    F: ArgminFloat,
{
    /// Constructor
    pub fn new(init_inverse_hessian: H, linesearch: L) -> Self {
        SR1 {
            r: F::from_f64(1e-8).unwrap(),
            init_inv_hessian: Some(init_inverse_hessian),
            linesearch,
            tol_grad: F::epsilon().sqrt(),
            tol_cost: F::epsilon(),
        }
    }

    /// Set r
    pub fn r(mut self, r: F) -> Result<Self, Error> {
        if r < F::from_f64(0.0).unwrap() || r > F::from_f64(1.0).unwrap() {
            Err(ArgminError::InvalidParameter {
                text: "SR1: r must be between 0 and 1.".to_string(),
            }
            .into())
        } else {
            self.r = r;
            Ok(self)
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

impl<O, L, P, G, H, F> Solver<O, IterState<P, G, (), H, F>> for SR1<L, H, F>
where
    O: CostFunction<Param = P, Output = F> + Gradient<Param = P, Gradient = G>,
    P: Clone
        + SerializeAlias
        + DeserializeOwnedAlias
        + ArgminSub<P, P>
        + ArgminDot<G, F>
        + ArgminDot<P, H>
        + ArgminNorm<F>
        + ArgminMul<F, P>,
    G: Clone + SerializeAlias + DeserializeOwnedAlias + ArgminNorm<F> + ArgminSub<G, G>,
    H: SerializeAlias + DeserializeOwnedAlias + ArgminDot<G, P> + ArgminAdd<H, H> + ArgminMul<F, H>,
    L: Clone + LineSearch<P, F> + Solver<O, IterState<P, G, (), (), F>>,
    F: ArgminFloat,
{
    const NAME: &'static str = "SR1";

    fn init(
        &mut self,
        op: &mut OpWrapper<O>,
        mut state: IterState<P, G, (), H, F>,
    ) -> Result<(IterState<P, G, (), H, F>, Option<ArgminKV>), Error> {
        let param = state.take_param().unwrap();
        let cost = op.cost(&param)?;
        let grad = op.gradient(&param)?;
        Ok((
            state
                .param(param)
                .cost(cost)
                .grad(grad)
                .inv_hessian(self.init_inv_hessian.take().unwrap()),
            None,
        ))
    }

    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        mut state: IterState<P, G, (), H, F>,
    ) -> Result<(IterState<P, G, (), H, F>, Option<ArgminKV>), Error> {
        let param = state.take_param().unwrap();
        let cost = state.cost;
        let mut inv_hessian = state.take_inv_hessian().unwrap();
        let prev_grad = state
            .take_grad()
            .map(Result::Ok)
            .unwrap_or_else(|| op.gradient(&param))?;

        let p = inv_hessian.dot(&prev_grad).mul(&F::from_f64(-1.0).unwrap());

        self.linesearch.set_search_direction(p);

        // Run solver
        let OptimizationResult {
            operator: line_op,
            state: mut linesearch_state,
        } = Executor::new(op.take_op().unwrap(), self.linesearch.clone())
            .configure(|config| {
                config
                    .param(param.clone())
                    .grad(prev_grad.clone())
                    .cost(cost)
            })
            .ctrlc(false)
            .run()?;

        let xk1 = linesearch_state.take_param().unwrap();
        let next_cost = linesearch_state.cost;

        // take care of function eval counts
        op.consume_op(line_op);

        let grad = op.gradient(&xk1)?;
        let yk = grad.sub(&prev_grad);

        let sk = xk1.sub(&param);

        let skmhkyk: P = sk.sub(&inv_hessian.dot(&yk));
        let a: H = skmhkyk.dot(&skmhkyk);
        let b: F = skmhkyk.dot(&yk);

        let hessian_update = b.abs() >= self.r * yk.norm() * skmhkyk.norm();

        // a try to see whether the skipping rule based on B_k makes any difference (seems not)
        // let bk = self.inv_hessian.inv()?;
        // let ykmbksk = yk.sub(&bk.dot(&sk));
        // let tmp: f64 = sk.dot(&ykmbksk);
        // let sksk: f64 = sk.dot(&sk);
        // let blah: f64 = ykmbksk.dot(&ykmbksk);
        // let hessian_update = tmp.abs() >= self.r * sksk.sqrt() * blah.sqrt();

        if hessian_update {
            inv_hessian = inv_hessian.add(&a.mul(&(F::from_f64(1.0).unwrap() / b)));
        }

        Ok((
            state
                .param(xk1)
                .cost(next_cost)
                .grad(grad)
                .inv_hessian(inv_hessian),
            Some(make_kv!["denom" => b; "hessian_update" => hessian_update;]),
        ))
    }

    fn terminate(&mut self, state: &IterState<P, G, (), H, F>) -> TerminationReason {
        if state.get_grad_ref().unwrap().norm() < self.tol_grad {
            return TerminationReason::TargetPrecisionReached;
        }
        if (state.get_prev_cost() - state.cost).abs() < self.tol_cost {
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
        sr1,
        SR1<MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64>, Vec<Vec<f64>>, f64>
    );

    #[test]
    fn test_tolerances() {
        let linesearch: MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64> =
            MoreThuenteLineSearch::new().c(1e-4, 0.9).unwrap();
        let init_hessian: Vec<Vec<f64>> = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        let tol1: f64 = 1e-4;
        let tol2: f64 = 1e-2;

        let SR1 {
            tol_grad: t1,
            tol_cost: t2,
            ..
        } = SR1::new(init_hessian, linesearch)
            .with_tol_grad(tol1)
            .with_tol_cost(tol2);

        assert!((t1 - tol1).abs() < std::f64::EPSILON);
        assert!((t2 - tol2).abs() < std::f64::EPSILON);
    }
}
