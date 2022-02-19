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
    ArgminFloat, ArgminIterData, ArgminLineSearch, ArgminOp, ArgminResult, Error, Executor,
    IterState, OpWrapper, SerializeAlias, Solver, State, TerminationReason,
};
use argmin_math::{
    ArgminAdd, ArgminDot, ArgminEye, ArgminMul, ArgminNorm, ArgminSub, ArgminTranspose,
};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// BFGS method
///
/// # References:
///
/// \[0\] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct BFGS<L, H, F> {
    /// Inverse Hessian
    init_inv_hessian: Option<H>,
    /// line search
    linesearch: L,
    /// Tolerance for the stopping criterion based on the change of the norm on the gradient
    tol_grad: F,
    /// Tolerance for the stopping criterion based on the change of the cost stopping criterion
    tol_cost: F,
}

impl<L, H, F> BFGS<L, H, F>
where
    F: ArgminFloat,
{
    /// Constructor
    pub fn new(init_inverse_hessian: H, linesearch: L) -> Self {
        BFGS {
            init_inv_hessian: Some(init_inverse_hessian),
            linesearch,
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

impl<O, L, H, F> Solver<IterState<O>> for BFGS<L, H, F>
where
    O: ArgminOp<Output = F, Hessian = H, Float = F>,
    O::Param: ArgminSub<O::Param, O::Param>
        + ArgminDot<O::Param, O::Float>
        + ArgminDot<O::Param, O::Hessian>
        + ArgminNorm<O::Float>
        + ArgminMul<O::Float, O::Param>,
    O::Hessian: SerializeAlias
        + ArgminSub<O::Hessian, O::Hessian>
        + ArgminDot<O::Param, O::Param>
        + ArgminDot<O::Hessian, O::Hessian>
        + ArgminAdd<O::Hessian, O::Hessian>
        + ArgminMul<O::Float, O::Hessian>
        + ArgminTranspose<O::Hessian>
        + ArgminEye,
    L: Clone + ArgminLineSearch<O::Param, O::Float> + Solver<IterState<O>>,
    F: ArgminFloat,
{
    const NAME: &'static str = "BFGS";

    fn init(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &mut IterState<O>,
    ) -> Result<Option<ArgminIterData<IterState<O>>>, Error> {
        let param = state.take_param().unwrap();
        let cost = op.apply(&param)?;
        let grad = op.gradient(&param)?;
        Ok(Some(
            ArgminIterData::new()
                .param(param)
                .cost(cost)
                .grad(grad)
                .inv_hessian(self.init_inv_hessian.take().unwrap()),
        ))
    }

    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &mut IterState<O>,
    ) -> Result<ArgminIterData<IterState<O>>, Error> {
        let param = state.take_param().unwrap();
        let cur_cost = state.get_cost();
        let prev_grad = state.take_grad().unwrap();
        let inv_hessian = state.get_inv_hessian().unwrap();

        let p = inv_hessian.dot(&prev_grad).mul(&F::from_f64(-1.0).unwrap());

        self.linesearch.set_search_direction(p);

        // Run solver
        let ArgminResult {
            operator: line_op,
            state: mut sub_state,
        } = Executor::new(
            op.take_op().unwrap(),
            self.linesearch.clone(),
            param.clone(),
        )
        .configure(|config| config.grad(prev_grad.clone()).cost(cur_cost))
        .ctrlc(false)
        .run()?;

        let xk1 = sub_state.take_param().unwrap();
        let next_cost = sub_state.get_cost();

        // take care of function eval counts
        op.consume_op(line_op);

        let grad = op.gradient(&xk1)?;

        let yk = grad.sub(&prev_grad);

        let sk = xk1.sub(&param);

        let yksk: F = yk.dot(&sk);
        let rhok = F::from_f64(1.0).unwrap() / yksk;

        let e = inv_hessian.eye_like();
        let mat1: O::Hessian = sk.dot(&yk);
        let mat1 = mat1.mul(&rhok);

        let tmp1 = e.sub(&mat1);

        let mat2 = mat1.t();
        let tmp2 = e.sub(&mat2);

        let sksk: O::Hessian = sk.dot(&sk);
        let sksk = sksk.mul(&rhok);

        // if state.get_iter() == 0 {
        //     let ykyk: f64 = yk.dot(&yk);
        //     self.inv_hessian = self.inv_hessian.eye_like().mul(&(yksk / ykyk));
        //     println!("{:?}", self.inv_hessian);
        // }

        let inv_hessian = tmp1.dot(&inv_hessian.dot(&tmp2)).add(&sksk);

        Ok(ArgminIterData::new()
            .param(xk1)
            .cost(next_cost)
            .grad(grad)
            .inv_hessian(inv_hessian))
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

    test_trait_impl!(bfgs, BFGS<Operator, MoreThuenteLineSearch<Operator, f64>, f64>);

    #[test]
    fn test_tolerances() {
        let linesearch: MoreThuenteLineSearch<f64, f64> =
            MoreThuenteLineSearch::new().c(1e-4, 0.9).unwrap();
        let init_hessian: Vec<Vec<f64>> = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        let tol1: f64 = 1e-4;
        let tol2: f64 = 1e-2;

        let BFGS {
            tol_grad: t1,
            tol_cost: t2,
            ..
        } = BFGS::new(init_hessian, linesearch)
            .with_tol_grad(tol1)
            .with_tol_cost(tol2);

        assert!((t1 - tol1).abs() < std::f64::EPSILON);
        assert!((t2 - tol2).abs() < std::f64::EPSILON);
    }
}
