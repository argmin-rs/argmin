// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Nelder Mead method
//!
//! # References:
//!
//! [Wikipedia](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method)

use crate::core::{
    ArgminError, ArgminFloat, CostFunction, Error, IterState, Problem, SerializeAlias, Solver,
    TerminationReason, KV,
};
use argmin_math::{ArgminAdd, ArgminMul, ArgminSub};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use std::fmt;

/// Nelder-Mead method
///
/// The Nelder-Mead method a heuristic search method for nonlinear optimization problems which does
/// not require derivatives.
///
/// The method is based on simplices which consist of n+1 vertices for an optimization problem with
/// n dimensions.
/// The function to be optimized is evaluated at all vertices. Based on these cost function values
/// the behaviour of the cost function is extrapolated in order to find the next point to be
/// evaluated.
///
/// The following actions are possible:
///
/// 1) Reflection: (Parameter `alpha`, default `1`)
/// 2) Expansion: (Parameter `gamma`, default `2`)
/// 3) Contraction: (Parameter `rho`, default `0.5`)
/// 4) Shrink: (Parameter `sigma`, default `0.5`)
///
/// # References:
///
/// [Wikipedia](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method)
#[derive(Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct NelderMead<P, F> {
    /// alpha
    alpha: F,
    /// gamma
    gamma: F,
    /// rho
    rho: F,
    /// sigma
    sigma: F,
    /// parameters
    params: Vec<(P, F)>,
    /// Sample standard deviation tolerance
    sd_tolerance: F,
}

impl<P, F> NelderMead<P, F>
where
    P: Clone + ArgminAdd<P, P> + ArgminSub<P, P> + ArgminMul<F, P>,
    F: ArgminFloat,
{
    /// Constructor
    pub fn new() -> Self {
        NelderMead {
            alpha: F::from_f64(1.0).unwrap(),
            gamma: F::from_f64(2.0).unwrap(),
            rho: F::from_f64(0.5).unwrap(),
            sigma: F::from_f64(0.5).unwrap(),
            params: vec![],
            sd_tolerance: F::epsilon(),
        }
    }

    /// Add initial parameters
    #[must_use]
    pub fn with_initial_params(mut self, params: Vec<P>) -> Self {
        self.params = params.into_iter().map(|p| (p, F::nan())).collect();
        self
    }

    /// Set Sample standard deviation tolerance
    #[must_use]
    pub fn sd_tolerance(mut self, tol: F) -> Self {
        self.sd_tolerance = tol;
        self
    }

    /// set alpha
    pub fn alpha(mut self, alpha: F) -> Result<Self, Error> {
        if alpha <= F::from_f64(0.0).unwrap() {
            return Err(ArgminError::InvalidParameter {
                text: "Nelder-Mead:  must be > 0.".to_string(),
            }
            .into());
        }
        self.alpha = alpha;
        Ok(self)
    }

    /// set gamma
    pub fn gamma(mut self, gamma: F) -> Result<Self, Error> {
        if gamma <= F::from_f64(1.0).unwrap() {
            return Err(ArgminError::InvalidParameter {
                text: "Nelder-Mead: gamma must be > 1.".to_string(),
            }
            .into());
        }
        self.gamma = gamma;
        Ok(self)
    }

    /// set rho
    pub fn rho(mut self, rho: F) -> Result<Self, Error> {
        if rho <= F::from_f64(0.0).unwrap() || rho > F::from_f64(0.5).unwrap() {
            return Err(ArgminError::InvalidParameter {
                text: "Nelder-Mead: rho must be in  (0.0, 0.5].".to_string(),
            }
            .into());
        }
        self.rho = rho;
        Ok(self)
    }

    /// set sigma
    pub fn sigma(mut self, sigma: F) -> Result<Self, Error> {
        if sigma <= F::from_f64(0.0).unwrap() || sigma > F::from_f64(1.0).unwrap() {
            return Err(ArgminError::InvalidParameter {
                text: "Nelder-Mead: sigma must be in  (0.0, 1.0].".to_string(),
            }
            .into());
        }
        self.sigma = sigma;
        Ok(self)
    }

    /// Sort parameters vectors based on their cost function values
    fn sort_param_vecs(&mut self) {
        self.params
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    }

    /// Calculate centroid of all but the worst vectors
    fn calculate_centroid(&self) -> P {
        let num_param = self.params.len() - 1;
        let mut x0: P = self.params[0].0.clone();
        for idx in 1..num_param {
            x0 = x0.add(&self.params[idx].0)
        }
        x0.mul(&(F::from_f64(1.0).unwrap() / (F::from_usize(num_param).unwrap())))
    }

    /// Reflect
    fn reflect(&self, x0: &P, x: &P) -> P {
        x0.add(&x0.sub(x).mul(&self.alpha))
    }

    /// Expand
    fn expand(&self, x0: &P, x: &P) -> P {
        x0.add(&x.sub(x0).mul(&self.gamma))
    }

    /// Contract
    fn contract(&self, x0: &P, x: &P) -> P {
        x0.add(&x.sub(x0).mul(&self.rho))
    }

    /// Shrink
    fn shrink<S>(&mut self, mut cost: S) -> Result<(), Error>
    where
        S: FnMut(&P) -> Result<F, Error>,
    {
        let mut out = Vec::with_capacity(self.params.len());
        out.push(self.params[0].clone());

        for idx in 1..self.params.len() {
            let xi = out[0]
                .0
                .add(&self.params[idx].0.sub(&out[0].0).mul(&self.sigma));
            let ci = (cost)(&xi)?;
            out.push((xi, ci));
        }
        self.params = out;
        Ok(())
    }
}

impl<P, F> Default for NelderMead<P, F>
where
    P: Clone + ArgminAdd<P, P> + ArgminSub<P, P> + ArgminMul<F, P>,
    F: ArgminFloat,
{
    fn default() -> NelderMead<P, F> {
        NelderMead::new()
    }
}

#[derive(Debug)]
enum Action {
    Reflection,
    Expansion,
    Contraction,
    Shrink,
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Action::Reflection => write!(f, "Reflection"),
            Action::Expansion => write!(f, "Expansion"),
            Action::Contraction => write!(f, "Contraction"),
            Action::Shrink => write!(f, "Shrink"),
        }
    }
}

impl<O, P, F> Solver<O, IterState<P, (), (), (), F>> for NelderMead<P, F>
where
    O: CostFunction<Param = P, Output = F>,
    P: Clone + SerializeAlias + ArgminSub<P, P> + ArgminAdd<P, P> + ArgminMul<F, P>,
    F: ArgminFloat + std::iter::Sum<F>,
{
    const NAME: &'static str = "Nelder-Mead method";

    fn init(
        &mut self,
        problem: &mut Problem<O>,
        state: IterState<P, (), (), (), F>,
    ) -> Result<(IterState<P, (), (), (), F>, Option<KV>), Error> {
        self.params = self
            .params
            .iter()
            .cloned()
            .map(|(p, _)| {
                let c = problem.cost(&p).unwrap();
                (p, c)
            })
            .collect();
        self.sort_param_vecs();

        Ok((
            state.param(self.params[0].0.clone()).cost(self.params[0].1),
            None,
        ))
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        state: IterState<P, (), (), (), F>,
    ) -> Result<(IterState<P, (), (), (), F>, Option<KV>), Error> {
        let num_param = self.params.len();

        let x0 = self.calculate_centroid();

        let xr = self.reflect(&x0, &self.params[num_param - 1].0);
        let xr_cost = problem.cost(&xr)?;

        let action = if xr_cost < self.params[num_param - 2].1 && xr_cost >= self.params[0].1 {
            // reflection
            self.params.last_mut().unwrap().0 = xr;
            self.params.last_mut().unwrap().1 = xr_cost;
            Action::Reflection
        } else if xr_cost < self.params[0].1 {
            // expansion
            let xe = self.expand(&x0, &xr);
            let xe_cost = problem.cost(&xe)?;
            if xe_cost < xr_cost {
                self.params.last_mut().unwrap().0 = xe;
                self.params.last_mut().unwrap().1 = xe_cost;
            } else {
                self.params.last_mut().unwrap().0 = xr;
                self.params.last_mut().unwrap().1 = xr_cost;
            }
            Action::Expansion
        } else if xr_cost >= self.params[num_param - 2].1 {
            // contraction
            let xc = self.contract(&x0, &self.params[num_param - 1].0);
            let xc_cost = problem.cost(&xc)?;
            if xc_cost < self.params[num_param - 1].1 {
                self.params.last_mut().unwrap().0 = xc;
                self.params.last_mut().unwrap().1 = xc_cost;
            }
            Action::Contraction
        } else {
            // shrink
            self.shrink(|x| problem.cost(x))?;
            Action::Shrink
        };

        self.sort_param_vecs();

        Ok((
            state.param(self.params[0].0.clone()).cost(self.params[0].1),
            Some(make_kv!("action" => action;)),
        ))
    }

    fn terminate(&mut self, _state: &IterState<P, (), (), (), F>) -> TerminationReason {
        let n = F::from_usize(self.params.len()).unwrap();
        let c0: F = self.params.iter().map(|(_, c)| *c).sum::<F>() / n;
        let s: F = (F::from_f64(1.0).unwrap() / (n - F::from_f64(1.0).unwrap())
            * self
                .params
                .iter()
                .map(|(_, c)| (*c - c0).powi(2))
                .sum::<F>())
        .sqrt();
        if s < self.sd_tolerance {
            return TerminationReason::TargetToleranceReached;
        }
        TerminationReason::NotTerminated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::PseudoOperator;
    use crate::test_trait_impl;

    test_trait_impl!(nelder_mead, NelderMead<PseudoOperator, f64>);
}
