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
    ArgminFloat, ArgminIterData, ArgminKV, ArgminOp, Error, IterState, OpWrapper, SerializeAlias,
    Solver,
};
use argmin_math::{ArgminConj, ArgminDot, ArgminMul, ArgminNorm, ArgminScaledAdd, ArgminSub};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// The conjugate gradient method is a solver for systems of linear equations with a symmetric and
/// positive-definite matrix.
///
/// # References:
///
/// \[0\] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct ConjugateGradient<P, F> {
    /// b (right hand side)
    b: P,
    /// residual
    r: Option<P>,
    /// p
    p: Option<P>,
    /// previous p
    p_prev: Option<P>,
    /// r^T * r
    #[cfg_attr(feature = "serde1", serde(skip))]
    rtr: F,
}

impl<P, F> ConjugateGradient<P, F>
where
    P: Clone,
    F: ArgminFloat,
{
    /// Constructor
    ///
    /// Parameters:
    ///
    /// `b`: right hand side of `A * x = b`
    pub fn new(b: P) -> Result<Self, Error> {
        Ok(ConjugateGradient {
            b,
            r: None,
            p: None,
            p_prev: None,
            rtr: F::nan(),
        })
    }

    /// Return the current search direction (This is needed by NewtonCG for instance)
    pub fn p(&self) -> &P {
        self.p.as_ref().unwrap()
    }

    /// Return the previous search direction (This is needed by NewtonCG for instance)
    pub fn p_prev(&self) -> &P {
        self.p_prev.as_ref().unwrap()
    }

    /// Return the current residual (This is needed by NewtonCG for instance)
    pub fn residual(&self) -> &P {
        self.r.as_ref().unwrap()
    }
}

impl<P, O, F> Solver<O> for ConjugateGradient<P, F>
where
    O: ArgminOp<Param = P, Output = P, Float = F>,
    P: Clone
        + SerializeAlias
        + ArgminDot<O::Param, F>
        + ArgminSub<O::Param, O::Param>
        + ArgminScaledAdd<O::Param, F, O::Param>
        + ArgminConj
        + ArgminMul<O::Float, O::Param>,
    F: ArgminFloat + ArgminNorm<O::Float>,
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
        self.r = Some(r0.clone());
        self.p = Some(r0.mul(&(F::from_f64(-1.0).unwrap())));
        self.rtr = r0.dot(&r0.conj());
        Ok(None)
    }

    /// Perform one iteration of CG algorithm
    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &IterState<O>,
    ) -> Result<ArgminIterData<O>, Error> {
        let p = self.p.as_ref().unwrap();
        let r = self.r.as_ref().unwrap();

        self.p_prev = Some(p.clone());
        let apk = op.apply(p)?;
        let alpha = self.rtr.div(p.dot(&apk.conj()));
        let new_param = state.get_param().scaled_add(&alpha, p);
        let r = r.scaled_add(&alpha, &apk);
        let rtr_n = r.dot(&r.conj());
        let beta = rtr_n.div(self.rtr);
        self.rtr = rtr_n;
        let p = r.mul(&(F::from_f64(-1.0).unwrap())).scaled_add(&beta, p);
        let norm = r.dot(&r.conj());

        self.p = Some(p);
        self.r = Some(r);

        Ok(ArgminIterData::new()
            .param(new_param)
            .cost(norm.norm())
            .kv(make_kv!("alpha" => alpha; "beta" => beta;)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_trait_impl;

    test_trait_impl!(conjugate_gradient, ConjugateGradient<Vec<f64>, f64>);
}
