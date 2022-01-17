// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # References:
//!
//! \[0\] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
//! Springer. ISBN 0-387-30303-0.

use crate::prelude::*;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use std::default::Default;
use std::fmt::Debug;

/// The conjugate gradient method is a solver for systems of linear equations with a symmetric and
/// positive-definite matrix.
///
/// # References:
///
/// \[0\] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct ConjugateGradient<P, S> {
    /// b (right hand side)
    b: P,
    /// residual
    r: P,
    /// p
    p: P,
    /// previous p
    p_prev: P,
    /// r^T * r
    #[cfg_attr(feature = "serde1", serde(skip))]
    rtr: S,
    /// alpha
    #[cfg_attr(feature = "serde1", serde(skip))]
    alpha: S,
    /// beta
    #[cfg_attr(feature = "serde1", serde(skip))]
    beta: S,
}

impl<P, S> ConjugateGradient<P, S>
where
    P: Clone + Default,
    S: Default,
{
    /// Constructor
    ///
    /// Parameters:
    ///
    /// `b`: right hand side of `A * x = b`
    pub fn new(b: P) -> Result<Self, Error> {
        Ok(ConjugateGradient {
            b,
            r: P::default(),
            p: P::default(),
            p_prev: P::default(),
            rtr: S::default(),
            alpha: S::default(),
            beta: S::default(),
        })
    }

    /// Return the current search direction (This is needed by NewtonCG for instance)
    pub fn p(&self) -> P {
        self.p.clone()
    }

    /// Return the previous search direction (This is needed by NewtonCG for instance)
    pub fn p_prev(&self) -> P {
        self.p_prev.clone()
    }

    /// Return the current residual (This is needed by NewtonCG for instance)
    pub fn residual(&self) -> P {
        self.r.clone()
    }
}

impl<P, O, S, F> Solver<O> for ConjugateGradient<P, S>
where
    O: ArgminOp<Param = P, Output = P, Float = F>,
    P: Clone
        + SerializeAlias
        + DeserializeOwnedAlias
        + ArgminDot<O::Param, S>
        + ArgminSub<O::Param, O::Param>
        + ArgminScaledAdd<O::Param, S, O::Param>
        + ArgminAdd<O::Param, O::Param>
        + ArgminConj
        + ArgminMul<O::Float, O::Param>,
    S: Debug + ArgminDiv<S, S> + ArgminNorm<O::Float> + ArgminConj,
    F: ArgminFloat,
{
    const NAME: &'static str = "Conjugate Gradient";

    fn init(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &IterState<O>,
    ) -> Result<Option<ArgminIterData<O>>, Error> {
        let init_param = state.get_param();
        let ap = op.apply(&init_param)?;
        let r0 = self.b.sub(&ap).mul(&(F::from_f64(-1.0).unwrap()));
        self.r = r0.clone();
        self.p = r0.mul(&(F::from_f64(-1.0).unwrap()));
        self.rtr = self.r.dot(&self.r.conj());
        Ok(None)
    }

    /// Perform one iteration of CG algorithm
    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &IterState<O>,
    ) -> Result<ArgminIterData<O>, Error> {
        self.p_prev = self.p.clone();
        let apk = op.apply(&self.p)?;
        self.alpha = self.rtr.div(&self.p.dot(&apk.conj()));
        let new_param = state.get_param().scaled_add(&self.alpha, &self.p);
        self.r = self.r.scaled_add(&self.alpha, &apk);
        let rtr_n = self.r.dot(&self.r.conj());
        self.beta = rtr_n.div(&self.rtr);
        self.rtr = rtr_n;
        self.p = self
            .r
            .mul(&(F::from_f64(-1.0).unwrap()))
            .scaled_add(&self.beta, &self.p);
        let norm = self.r.dot(&self.r.conj());

        Ok(ArgminIterData::new()
            .param(new_param)
            // .cost(norm.sqrt())
            .cost(norm.norm())
            .kv(make_kv!("alpha" => self.alpha; "beta" => self.beta;)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_trait_impl;

    test_trait_impl!(conjugate_gradient, ConjugateGradient<Vec<f64>, f64>);
}
