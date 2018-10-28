// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Nonlinear Conjugate Gradient Method
//!
//! TODO: Proper documentation.
//!
// //!
// //! # Example
// //!
// //! ```rust
// //! todo
// //! ```

use prelude::*;
use std;
use std::default::Default;

/// Nonlinear Conjugate Gradient struct
#[derive(ArgminSolver)]
pub struct NonlinearConjugateGradient<'a, T, H>
where
    T: Clone
        + Default
        + ArgminSub<T>
        + ArgminAdd<T>
        + ArgminScale<f64>
        + ArgminDot<T, f64>
        + ArgminScaledAdd<T, f64>
        + ArgminScaledSub<T, f64>,
    H: Clone + Default,
{
    /// b
    b: T,
    /// residual
    r: T,
    /// p
    p: T,
    /// r^T * r
    rtr: f64,
    /// alpha
    alpha: f64,
    /// beta
    beta: f64,
    /// base
    base: ArgminBase<'a, T, T, H>,
}

impl<'a, T, H> NonlinearConjugateGradient<'a, T, H>
where
    T: Clone
        + Default
        + ArgminSub<T>
        + ArgminAdd<T>
        + ArgminScale<f64>
        + ArgminDot<T, f64>
        + ArgminScaledAdd<T, f64>
        + ArgminScaledSub<T, f64>,
    H: Clone + Default,
{
    /// Constructor
    ///
    /// Parameters:
    ///
    /// `cost_function`: cost function
    /// `init_param`: Initial parameter vector
    pub fn new(
        operator: Box<ArgminOperator<Parameters = T, OperatorOutput = T, Hessian = H>>,
        b: T,
        init_param: T,
    ) -> Result<Self, Error> {
        unimplemented!()
        // Ok(NonlinearConjugateGradient {
        //     b: b,
        //     r: T::default(),
        //     p: T::default(),
        //     rtr: std::f64::NAN,
        //     alpha: std::f64::NAN,
        //     beta: std::f64::NAN,
        //     base: ArgminBase::new(operator, init_param),
        // })
    }
}

impl<'a, T, H> ArgminNextIter for NonlinearConjugateGradient<'a, T, H>
where
    T: Clone
        + Default
        + ArgminSub<T>
        + ArgminAdd<T>
        + ArgminScale<f64>
        + ArgminDot<T, f64>
        + ArgminScaledAdd<T, f64>
        + ArgminScaledSub<T, f64>,
    H: Clone + Default,
{
    type Parameters = T;
    type OperatorOutput = T;
    type Hessian = H;

    fn init(&mut self) -> Result<(), Error> {
        unimplemented!();
        // let init_param = self.cur_param();
        // let ap = self.apply(&init_param)?;
        // let r0 = self.b.sub(ap).scale(-1.0);
        // self.r = r0.clone();
        // self.p = r0.scale(-1.0);
        // self.rtr = self.r.dot(self.r.clone());
        Ok(())
    }

    /// Perform one iteration of SA algorithm
    fn next_iter(&mut self) -> Result<ArgminIterationData<Self::Parameters>, Error> {
        unimplemented!()
        // Still way too much cloning going on here
        // let p = self.p.clone();
        // let apk = self.apply(&p)?;
        // self.alpha = self.rtr / self.p.dot(apk.clone());
        // let new_param = self.cur_param().scaled_add(self.alpha, p.clone());
        // self.r = self.r.scaled_add(self.alpha, apk);
        // let rtr_n = self.r.dot(self.r.clone());
        // self.beta = rtr_n / self.rtr;
        // self.rtr = rtr_n;
        // self.p = self.r.scale(-1.0).scaled_add(self.beta, p);
        // let norm = self.r.dot(self.r.clone());
        //
        // let mut out = ArgminIterationData::new(new_param, norm);
        // out.add_kv(make_kv!(
        //         "alpha" => self.alpha;
        //         "beta" => self.beta;
        //     ));
        // Ok(out)
    }
}
