// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # References:
//!
//! [0] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
//! Springer. ISBN 0-387-30303-0.

use crate::prelude::*;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::default::Default;

/// The conjugate gradient method is a solver for systems of linear equations with a symmetric and
/// positive-definite matrix.
///
/// [Example](https://github.com/argmin-rs/argmin/blob/master/examples/conjugategradient.rs)
///
/// # References:
///
/// [0] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(Clone, Serialize, Deserialize)]
pub struct ConjugateGradient<P> {
    /// b (right hand side)
    b: P,
    /// residual
    r: P,
    /// p
    p: P,
    /// previous p
    p_prev: P,
    /// r^T * r
    rtr: f64,
    /// alpha
    alpha: f64,
    /// beta
    beta: f64,
}

impl<P> ConjugateGradient<P>
where
    P: Clone + Default,
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
            rtr: std::f64::NAN,
            alpha: std::f64::NAN,
            beta: std::f64::NAN,
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

impl<P, O> Solver<O> for ConjugateGradient<P>
where
    O: ArgminOp<Param = P, Output = P>,
    P: Clone
        + Serialize
        + DeserializeOwned
        + ArgminSub<P, P>
        + ArgminDot<P, f64>
        + ArgminScaledAdd<P, f64, P>
        + ArgminAdd<P, P>
        + ArgminMul<f64, P>
        + ArgminDot<P, f64>,
{
    const NAME: &'static str = "Conjugate Gradient";

    fn init(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &IterState<O>,
    ) -> Result<Option<ArgminIterData<O>>, Error> {
        let init_param = state.get_param();
        let ap = op.apply(&init_param)?;
        let r0 = self.b.sub(&ap).mul(&(-1.0));
        self.r = r0.clone();
        self.p = r0.mul(&(-1.0));
        self.rtr = self.r.dot(&self.r);
        Ok(None)
    }

    /// Perform one iteration of SA algorithm
    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &IterState<O>,
    ) -> Result<ArgminIterData<O>, Error> {
        self.p_prev = self.p.clone();
        let apk = op.apply(&self.p)?;
        self.alpha = self.rtr / self.p.dot(&apk);
        let new_param = state.get_param().scaled_add(&self.alpha, &self.p);
        self.r = self.r.scaled_add(&self.alpha, &apk);
        let rtr_n = self.r.dot(&self.r);
        self.beta = rtr_n / self.rtr;
        self.rtr = rtr_n;
        self.p = self.r.mul(&(-1.0)).scaled_add(&self.beta, &self.p);
        let norm = self.r.dot(&self.r);

        Ok(ArgminIterData::new()
            .param(new_param)
            .cost(norm.sqrt())
            .kv(make_kv!("alpha" => self.alpha; "beta" => self.beta;)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::send_sync_test;

    send_sync_test!(
        conjugate_gradient,
        ConjugateGradient<NoOperator<Vec<f64>, Vec<f64>, ()>>
    );
}
