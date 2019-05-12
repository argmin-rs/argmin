// Copyright 2018-2019 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # References:
//!
//! [Wikipedia](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method)

use crate::prelude::*;
use serde::{Deserialize, Serialize};
use std::default::Default;

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
/// [Example](https://github.com/argmin-rs/argmin/blob/master/examples/neldermead.rs)
///
/// # References:
///
/// [Wikipedia](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method)
#[derive(Serialize, Deserialize)]
pub struct NelderMead<O: ArgminOp> {
    /// alpha
    alpha: f64,
    /// gamma
    gamma: f64,
    /// rho
    rho: f64,
    /// sigma
    sigma: f64,
    /// parameters
    params: Vec<(O::Param, f64)>,
}

impl<O: ArgminOp> NelderMead<O>
where
    O: ArgminOp<Output = f64>,
    O::Param: Default
        + ArgminAdd<O::Param, O::Param>
        + ArgminSub<O::Param, O::Param>
        + ArgminMul<f64, O::Param>,
{
    /// Constructor
    pub fn new() -> Self {
        NelderMead {
            alpha: 1.0,
            gamma: 2.0,
            rho: 0.5,
            sigma: 0.5,
            params: vec![],
        }
    }

    /// set alpha
    pub fn alpha(mut self, alpha: f64) -> Result<Self, Error> {
        if alpha <= 0.0 {
            return Err(ArgminError::InvalidParameter {
                text: "Nelder-Mead:  must be > 0.".to_string(),
            }
            .into());
        }
        self.alpha = alpha;
        Ok(self)
    }

    /// set gamma
    pub fn gamma(mut self, gamma: f64) -> Result<Self, Error> {
        if gamma <= 1.0 {
            return Err(ArgminError::InvalidParameter {
                text: "Nelder-Mead: gamma must be > 1.".to_string(),
            }
            .into());
        }
        self.gamma = gamma;
        Ok(self)
    }

    /// set rho
    pub fn rho(mut self, rho: f64) -> Result<Self, Error> {
        if rho <= 0.0 || rho > 0.5 {
            return Err(ArgminError::InvalidParameter {
                text: "Nelder-Mead: rho must be in  (0.0, 0.5].".to_string(),
            }
            .into());
        }
        self.rho = rho;
        Ok(self)
    }

    /// set sigma
    pub fn sigma(mut self, sigma: f64) -> Result<Self, Error> {
        if sigma <= 0.0 || sigma > 1.0 {
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
    fn calculate_centroid(&self) -> O::Param {
        let num_param = self.params.len() - 1;
        let mut x0: O::Param = self.params[0].0.clone();
        for idx in 1..num_param {
            x0 = x0.add(&self.params[idx].0)
        }
        x0.mul(&(1.0 / (num_param as f64)))
    }

    /// Reflect
    fn reflect(&self, x0: &O::Param, x: &O::Param) -> O::Param {
        x0.add(&x0.sub(&x).mul(&self.alpha))
    }

    /// Expand
    fn expand(&self, x0: &O::Param, x: &O::Param) -> O::Param {
        x0.add(&x.sub(&x0).mul(&self.gamma))
    }

    /// Contract
    fn contract(&self, x0: &O::Param, x: &O::Param) -> O::Param {
        x0.add(&x.sub(&x0).mul(&self.rho))
    }

    /// Shrink
    fn shrink(&mut self, cost: &Fn(&O::Param) -> O::Output) {
        let mut out = Vec::with_capacity(self.params.len());
        out.push(self.params[0].clone());

        for idx in 1..self.params.len() {
            let xi = out[0]
                .0
                .add(&self.params[idx].0.sub(&out[0].0).mul(&self.sigma));
            let ci = (cost)(&xi);
            out.push((xi, ci));
        }
        self.params = out;
    }
}

// impl<O> Default for NelderMead<O> {
//     fn default() -> NelderMead<O> {
//         NelderMead::new()
//     }
// }

impl<O> Solver<O> for NelderMead<O>
where
    O: ArgminOp<Output = f64>,
    O::Param: Default
        + ArgminScaledSub<O::Param, f64, O::Param>
        + ArgminSub<O::Param, O::Param>
        + ArgminMul<f64, O::Param>,
{
    const NAME: &'static str = "Nelder-Mead method";

    fn init(
        &mut self,
        _op: &mut OpWrapper<O>,
        _state: &IterState<O>,
    ) -> Result<Option<ArgminIterData<O>>, Error> {
        unimplemented!()
    }

    fn next_iter(
        &mut self,
        _op: &mut OpWrapper<O>,
        _state: &IterState<O>,
    ) -> Result<ArgminIterData<O>, Error> {
        // let param = state.get_param();
        // let residuals = op.apply(&param)?;
        // let jacobian = op.jacobian(&param)?;
        //
        // Ok(ArgminIterData::new()
        //     .param(new_param)
        //     .cost(residuals.norm()))
        unimplemented!()
    }

    fn terminate(&mut self, _state: &IterState<O>) -> TerminationReason {
        // if (state.get_prev_cost() - state.get_cost()).abs() < std::f64::EPSILON.sqrt() {
        //     return TerminationReason::NoChangeInCost;
        // }
        TerminationReason::NotTerminated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::send_sync_test;

    send_sync_test!(nelder_mead, NelderMead);
}
