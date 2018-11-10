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
//! # References:
//!
//! [0] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
//! Springer. ISBN 0-387-30303-0.
// //!
// //! # Example
// //!
// //! ```rust
// //! todo
// //! ```

use prelude::*;
use std;
use std::default::Default;

/// Conjugate Gradient struct
#[derive(ArgminSolver)]
pub struct ConjugateGradient<'a, T>
where
    T: Clone
        + Default
        + ArgminSub<T>
        + ArgminAdd<T>
        + ArgminScale<f64>
        + ArgminDot<T, f64>
        + ArgminScaledAdd<T, f64>
        + ArgminScaledSub<T, f64>,
{
    /// b
    b: T,
    /// residual
    r: T,
    /// p
    p: T,
    /// previous p
    p_prev: T,
    /// r^T * r
    rtr: f64,
    /// alpha
    alpha: f64,
    /// beta
    beta: f64,
    /// base
    base: ArgminBase<'a, T, T, ()>,
}

impl<'a, T> ConjugateGradient<'a, T>
where
    T: Clone
        + Default
        + ArgminSub<T>
        + ArgminAdd<T>
        + ArgminScale<f64>
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
        operator: Box<ArgminOperator<Parameters = T, OperatorOutput = T, Hessian = ()> + 'a>,
        b: T,
        init_param: T,
    ) -> Result<Self, Error> {
        Ok(ConjugateGradient {
            b: b,
            r: T::default(),
            p: T::default(),
            p_prev: T::default(),
            rtr: std::f64::NAN,
            alpha: std::f64::NAN,
            beta: std::f64::NAN,
            base: ArgminBase::new(operator, init_param),
        })
    }

    /// Return the current search direction (This is needed by NewtonCG for instance)
    pub fn p(&self) -> T {
        self.p.clone()
    }

    /// Return the previous search direction (This is needed by NewtonCG for instance)
    pub fn p_prev(&self) -> T {
        self.p_prev.clone()
    }

    /// Return the current residual (This is needed by NewtonCG for instance)
    pub fn residual(&self) -> T {
        self.r.clone()
    }
}

impl<'a, T> ArgminNextIter for ConjugateGradient<'a, T>
where
    T: Clone
        + Default
        + ArgminSub<T>
        + ArgminAdd<T>
        + ArgminScale<f64>
        + ArgminDot<T, f64>
        + ArgminScaledAdd<T, f64>
        + ArgminScaledSub<T, f64>,
{
    type Parameters = T;
    type OperatorOutput = T;
    type Hessian = ();

    fn init(&mut self) -> Result<(), Error> {
        let init_param = self.cur_param();
        let ap = self.apply(&init_param)?;
        let r0 = self.b.sub(ap).scale(-1.0);
        self.r = r0.clone();
        self.p = r0.scale(-1.0);
        self.rtr = self.r.dot(self.r.clone());
        Ok(())
    }

    /// Perform one iteration of SA algorithm
    fn next_iter(&mut self) -> Result<ArgminIterationData<Self::Parameters>, Error> {
        // Still way too much cloning going on here
        self.p_prev = self.p.clone();
        let p = self.p.clone();
        let apk = self.apply(&p)?;
        self.alpha = self.rtr / self.p.dot(apk.clone());
        let new_param = self.cur_param().scaled_add(self.alpha, p.clone());
        self.r = self.r.scaled_add(self.alpha, apk);
        let rtr_n = self.r.dot(self.r.clone());
        self.beta = rtr_n / self.rtr;
        self.rtr = rtr_n;
        self.p = self.r.scale(-1.0).scaled_add(self.beta, p);
        let norm = self.r.dot(self.r.clone());

        let mut out = ArgminIterationData::new(new_param, norm.sqrt());
        out.add_kv(make_kv!(
                "alpha" => self.alpha;
                "beta" => self.beta;
            ));
        Ok(out)
    }
}
