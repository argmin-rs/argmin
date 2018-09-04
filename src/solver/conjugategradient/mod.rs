// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Conjugate Gradient Method
//!
//! TODO: Proper documentation.
//!
// //!
// //! # Example
// //!
// //! ```rust
// //! todo
// //! ```

use argmin_core::{
    ArgminBase, ArgminDot, ArgminIterationData, ArgminKV, ArgminLog, ArgminNextIter,
    ArgminOperator, ArgminResult, ArgminScaledAdd, ArgminScaledSub, ArgminSolver, ArgminSub,
    ArgminWrite, Error, TerminationReason,
};
use std;
use std::default::Default;

/// Conjugate Gradient struct
#[derive(ArgminSolver)]
pub struct ConjugateGradient<T>
where
    T: Clone
        + Default
        + ArgminSub<T>
        + ArgminDot<T, f64>
        + ArgminScaledAdd<T, f64>
        + ArgminScaledSub<T, f64>,
{
    /// residual
    r: T,
    /// p
    p: T,
    /// alpha
    alpha: f64,
    /// beta
    beta: f64,
    /// base
    base: ArgminBase<T, T>,
}

impl<T> ConjugateGradient<T>
where
    T: Clone
        + Default
        + ArgminSub<T>
        + ArgminDot<T, f64>
        + ArgminScaledAdd<T, f64>
        + ArgminScaledSub<T, f64>,
{
    /// Constructor
    ///
    /// Parameters:
    ///
    /// `cost_function`: cost function
    /// `init_param`: Initial parameter vector
    pub fn new(
        operator: Box<ArgminOperator<Parameters = T, OperatorOutput = T>>,
        b: T,
        init_param: T,
    ) -> Result<Self, Error> {
        let ap = operator.apply(&init_param)?;
        let r0: T = b.sub(ap);
        Ok(ConjugateGradient {
            r: r0.clone(),
            p: r0,
            alpha: std::f64::NAN,
            beta: std::f64::NAN,
            base: ArgminBase::new(operator, init_param),
        })
    }
}

impl<T> ArgminNextIter for ConjugateGradient<T>
where
    T: Clone
        + Default
        + ArgminSub<T>
        + ArgminDot<T, f64>
        + ArgminScaledAdd<T, f64>
        + ArgminScaledSub<T, f64>,
{
    type Parameters = T;
    type OperatorOutput = T;

    /// Perform one iteration of SA algorithm
    fn next_iter(&mut self) -> Result<ArgminIterationData<Self::Parameters>, Error> {
        // Still way too much cloning going on here
        let p = self.p.clone();
        let apk = self.apply(&p)?;
        let rtr = self.r.dot(self.r.clone());
        self.alpha = rtr / self.p.dot(apk.clone());
        let new_param = self.base.cur_param().scaled_add(self.alpha, p.clone());
        self.r = self.r.scaled_sub(self.alpha, apk);
        self.beta = self.r.dot(self.r.clone()) / rtr;
        self.p = self.r.scaled_add(self.beta, p);
        let norm = self.r.dot(self.r.clone());

        let mut out = ArgminIterationData::new(new_param, norm);
        out.add_kv(make_kv!(
                "alpha" => self.alpha;
                "beta" => self.beta;
            ));
        Ok(out)
    }
}
