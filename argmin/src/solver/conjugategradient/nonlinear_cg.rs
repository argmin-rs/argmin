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
    ArgminFloat, ArgminKV, CostFunction, DeserializeOwnedAlias, Error, Executor, Gradient,
    IterState, LineSearch, NLCGBetaUpdate, OpWrapper, OptimizationResult, SerializeAlias, Solver,
    State,
};
use argmin_math::{ArgminAdd, ArgminDot, ArgminMul, ArgminNorm};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// The nonlinear conjugate gradient is a generalization of the conjugate gradient method for
/// nonlinear optimization problems.
///
/// # References:
///
/// \[0\] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct NonlinearConjugateGradient<P, L, B, F> {
    /// p
    p: Option<P>,
    /// beta
    beta: F,
    /// line search
    linesearch: L,
    /// beta update method
    beta_method: B,
    /// Number of iterations after which a restart is performed
    restart_iter: u64,
    /// Restart based on orthogonality
    restart_orthogonality: Option<F>,
}

impl<P, L, B, F> NonlinearConjugateGradient<P, L, B, F>
where
    F: ArgminFloat,
{
    /// Constructor (Polak Ribiere Conjugate Gradient (PR-CG))
    pub fn new(linesearch: L, beta_method: B) -> Result<Self, Error> {
        Ok(NonlinearConjugateGradient {
            p: None,
            beta: F::nan(),
            linesearch,
            beta_method,
            restart_iter: std::u64::MAX,
            restart_orthogonality: None,
        })
    }

    /// Specifiy the number of iterations after which a restart should be performed
    /// This allows the algorithm to "forget" previous information which may not be helpful
    /// anymore.
    #[must_use]
    pub fn restart_iters(mut self, iters: u64) -> Self {
        self.restart_iter = iters;
        self
    }

    /// Set the value for the orthogonality measure.
    /// Setting this parameter leads to a restart of the algorithm (setting beta = 0) after two
    /// consecutive search directions are not orthogonal anymore. In other words, if this condition
    /// is met:
    ///
    /// `|\nabla f_k^T * \nabla f_{k-1}| / | \nabla f_k ||^2 >= v`
    ///
    /// A typical value for `v` is 0.1.
    #[must_use]
    pub fn restart_orthogonality(mut self, v: F) -> Self {
        self.restart_orthogonality = Some(v);
        self
    }
}

impl<O, P, G, L, B, F> Solver<O, IterState<P, G, (), (), F>>
    for NonlinearConjugateGradient<P, L, B, F>
where
    O: CostFunction<Param = P, Output = F> + Gradient<Param = P, Gradient = G>,
    P: Clone + SerializeAlias + DeserializeOwnedAlias + ArgminAdd<P, P> + ArgminMul<F, P>,
    G: Clone
        + SerializeAlias
        + DeserializeOwnedAlias
        + ArgminMul<F, P>
        + ArgminDot<G, F>
        + ArgminNorm<F>,
    L: Clone + LineSearch<P, F> + Solver<O, IterState<P, G, (), (), F>>,
    B: NLCGBetaUpdate<G, P, F>,
    F: ArgminFloat,
{
    const NAME: &'static str = "Nonlinear Conjugate Gradient";

    fn init(
        &mut self,
        op: &mut OpWrapper<O>,
        mut state: IterState<P, G, (), (), F>,
    ) -> Result<(IterState<P, G, (), (), F>, Option<ArgminKV>), Error> {
        let param = state.take_param().unwrap();
        let cost = op.cost(&param)?;
        let grad = op.gradient(&param)?;
        self.p = Some(grad.mul(&(F::from_f64(-1.0).unwrap())));
        Ok((state.param(param).cost(cost).grad(grad), None))
    }

    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        mut state: IterState<P, G, (), (), F>,
    ) -> Result<(IterState<P, G, (), (), F>, Option<ArgminKV>), Error> {
        let p = self.p.as_ref().unwrap();
        let xk = state.take_param().unwrap();
        let grad = state
            .take_grad()
            .map(Result::Ok)
            .unwrap_or_else(|| op.gradient(&xk))?;
        let cur_cost = state.cost;

        // Linesearch
        self.linesearch.set_search_direction(p.clone());

        // Run solver
        let OptimizationResult {
            operator: line_op,
            state: mut line_state,
        } = Executor::new(op.take_op().unwrap(), self.linesearch.clone())
            .configure(|config| config.param(xk).grad(grad.clone()).cost(cur_cost))
            .ctrlc(false)
            .run()?;

        // takes care of the counts of function evaluations
        op.consume_op(line_op);

        let xk1 = line_state.take_param().unwrap();

        // Update of beta
        let new_grad = op.gradient(&xk1)?;

        let restart_orthogonality = match self.restart_orthogonality {
            Some(v) => new_grad.dot(&grad).abs() / new_grad.norm().powi(2) >= v,
            None => false,
        };

        let restart_iter: bool =
            (state.get_iter() % self.restart_iter == 0) && state.get_iter() != 0;

        if restart_iter || restart_orthogonality {
            self.beta = F::from_f64(0.0).unwrap();
        } else {
            self.beta = self.beta_method.update(&grad, &new_grad, p);
        }

        // Update of p
        self.p = Some(
            new_grad
                .mul(&(F::from_f64(-1.0).unwrap()))
                .add(&p.mul(&self.beta)),
        );

        // Housekeeping
        let cost = op.cost(&xk1)?;

        Ok((
            state.param(xk1).cost(cost).grad(new_grad),
            Some(make_kv!("beta" => self.beta;
             "restart_iter" => restart_iter;
             "restart_orthogonality" => restart_orthogonality;
            )),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::MinimalNoOperator;
    use crate::solver::conjugategradient::beta::PolakRibiere;
    use crate::solver::linesearch::MoreThuenteLineSearch;
    use crate::test_trait_impl;

    test_trait_impl!(
        nonlinear_cg,
        NonlinearConjugateGradient<
            MinimalNoOperator,
            MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64>,
            PolakRibiere,
            f64
        >
    );
}
