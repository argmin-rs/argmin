// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! TODO: Stop when search direction is close to 0
//!
//! # References:
//!
//! [0] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
//! Springer. ISBN 0-387-30303-0.

use crate::prelude::*;
use crate::solver::conjugategradient::ConjugateGradient;
use serde::{Deserialize, Serialize};

/// The Newton-CG method (also called truncated Newton method) uses a modified CG to solve the
/// Newton equations approximately. After a search direction is found, a line search is performed.
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
#[derive(Serialize, Deserialize)]
pub struct NewtonCG<L> {
    /// line search
    linesearch: L,
    /// curvature_threshold
    curvature_threshold: f64,
}

impl<L> NewtonCG<L> {
    /// Constructor
    pub fn new(linesearch: L) -> Self {
        NewtonCG {
            linesearch,
            curvature_threshold: 0.0,
        }
    }

    /// Set curvature threshold
    pub fn curvature_threshold(mut self, threshold: f64) -> Self {
        self.curvature_threshold = threshold;
        self
    }
}

impl<O, L> Solver<O> for NewtonCG<L>
where
    O: ArgminOp<Output = f64>,
    O::Param: Send
        + Sync
        + Clone
        + Serialize
        + Default
        + ArgminSub<O::Param, O::Param>
        + ArgminAdd<O::Param, O::Param>
        + ArgminDot<O::Param, f64>
        + ArgminScaledAdd<O::Param, f64, O::Param>
        + ArgminMul<f64, O::Param>
        + ArgminZeroLike
        + ArgminNorm<f64>,
    O::Hessian: Send
        + Sync
        + Clone
        + Serialize
        + Default
        + ArgminInv<O::Hessian>
        + ArgminDot<O::Param, O::Param>,
    L: Clone + ArgminLineSearch<O::Param> + Solver<OpWrapper<O>>,
{
    const NAME: &'static str = "Newton-CG";

    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &IterState<O>,
    ) -> Result<ArgminIterData<O>, Error> {
        let param = state.get_param();
        let grad = op.gradient(&param)?;
        let hessian = op.hessian(&param)?;

        // Solve CG subproblem
        let cg_op: CGSubProblem<O::Param, O::Hessian> = CGSubProblem::new(hessian.clone());
        let mut cg_op = OpWrapper::new(&cg_op);

        let mut x_p = param.zero_like();
        let mut x: O::Param = param.zero_like();
        let mut cg = ConjugateGradient::new(grad.mul(&(-1.0)))?;

        let mut cg_state = IterState::new(x_p.clone());
        cg.init(&mut cg_op, &cg_state)?;
        let grad_norm = grad.norm();
        for iter in 0.. {
            let data = cg.next_iter(&mut cg_op, &cg_state)?;
            x = data.get_param().unwrap();
            let p = cg.p_prev();
            let curvature = p.dot(&hessian.dot(&p));
            if curvature <= self.curvature_threshold {
                if iter == 0 {
                    x = grad.mul(&(-1.0));
                    break;
                } else {
                    x = x_p;
                    break;
                }
            }
            if data.get_cost().unwrap() <= (0.5f64).min(grad_norm.sqrt()) * grad_norm {
                break;
            }
            cg_state.param(x.clone());
            cg_state.cost(data.get_cost().unwrap());
            x_p = x.clone();
        }

        // take care of counting
        op.consume_op(cg_op);

        // perform line search
        self.linesearch.set_search_direction(x);

        // Run solver
        let ArgminResult {
            operator: line_op,
            state:
                IterState {
                    param: next_param,
                    cost: next_cost,
                    ..
                },
        } = Executor::new(OpWrapper::new_from_op(&op), self.linesearch.clone(), param)
            .grad(grad)
            .cost(state.get_cost())
            .ctrlc(false)
            .run()?;

        op.consume_op(line_op);

        Ok(ArgminIterData::new().param(next_param).cost(next_cost))
    }

    fn terminate(&mut self, state: &IterState<O>) -> TerminationReason {
        if (state.get_cost() - state.get_prev_cost()).abs() < std::f64::EPSILON {
            TerminationReason::NoChangeInCost
        } else {
            TerminationReason::NotTerminated
        }
    }
}

#[derive(Clone, Default, Serialize, Deserialize)]
struct CGSubProblem<T, H> {
    hessian: H,
    phantom: std::marker::PhantomData<T>,
}

impl<T, H> CGSubProblem<T, H>
where
    T: Clone + Send + Sync,
    H: Clone + Default + ArgminDot<T, T> + Send + Sync,
{
    /// constructor
    pub fn new(hessian: H) -> Self {
        CGSubProblem {
            hessian,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<T, H> ArgminOp for CGSubProblem<T, H>
where
    T: Clone + Default + Send + Sync + Serialize + serde::de::DeserializeOwned,
    H: Clone + Default + ArgminDot<T, T> + Send + Sync + Serialize + serde::de::DeserializeOwned,
{
    type Param = T;
    type Output = T;
    type Hessian = ();

    fn apply(&self, p: &T) -> Result<T, Error> {
        Ok(self.hessian.dot(&p))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::send_sync_test;
    use crate::solver::linesearch::MoreThuenteLineSearch;

    // Only works with ndarray feature because of the required inverse of a matrix
    #[cfg(feature = "ndarrayl")]
    type Operator = NoOperator<ndarray::Array1<f64>, f64, ndarray::Array2<f64>>;

    // Only works with ndarray feature because of the required inverse of a matrix
    #[cfg(feature = "ndarrayl")]
    send_sync_test!(newton_cg, NewtonCG<Operator, MoreThuenteLineSearch<Operator>>);

    send_sync_test!(cg_subproblem, CGSubProblem<Vec<f64>, Vec<Vec<f64>>>);
}
