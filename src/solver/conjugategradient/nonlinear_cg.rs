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
use solver::linesearch::HagerZhangLineSearch;
use std;
use std::default::Default;

/// Nonlinear Conjugate Gradient struct
#[derive(ArgminSolver)]
pub struct NonlinearConjugateGradient<'a, T>
where
    T: 'a
        + Clone
        + Default
        + ArgminSub<T>
        + ArgminAdd<T>
        + ArgminScale<f64>
        + ArgminNorm<f64>
        + ArgminDot<T, f64>
        + ArgminScaledAdd<T, f64>
        + ArgminScaledSub<T, f64>,
{
    /// p
    p: T,
    /// alpha
    alpha: f64,
    /// beta
    beta: f64,
    /// line search
    linesearch: Box<ArgminLineSearch<Parameters = T, OperatorOutput = f64, Hessian = ()> + 'a>,
    /// base
    base: ArgminBase<'a, T, f64, ()>,
}

impl<'a, T> NonlinearConjugateGradient<'a, T>
where
    T: 'a
        + Clone
        + Default
        + ArgminSub<T>
        + ArgminAdd<T>
        + ArgminScale<f64>
        + ArgminNorm<f64>
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
        operator: Box<ArgminOperator<Parameters = T, OperatorOutput = f64, Hessian = ()>>,
        init_param: T,
    ) -> Result<Self, Error> {
        let linesearch = HagerZhangLineSearch::new(operator.clone());
        Ok(NonlinearConjugateGradient {
            p: T::default(),
            alpha: std::f64::NAN,
            beta: std::f64::NAN,
            linesearch: Box::new(linesearch),
            base: ArgminBase::new(operator, init_param),
        })
    }

    /// Specify line search method
    pub fn set_linesearch(
        &mut self,
        linesearch: Box<ArgminLineSearch<Parameters = T, OperatorOutput = f64, Hessian = ()> + 'a>,
    ) -> &mut Self {
        self.linesearch = linesearch;
        self
    }
}

impl<'a, T> ArgminNextIter for NonlinearConjugateGradient<'a, T>
where
    T: 'a
        + Clone
        + Default
        + ArgminSub<T>
        + ArgminAdd<T>
        + ArgminScale<f64>
        + ArgminNorm<f64>
        + ArgminDot<T, f64>
        + ArgminScaledAdd<T, f64>
        + ArgminScaledSub<T, f64>,
{
    type Parameters = T;
    type OperatorOutput = f64;
    type Hessian = ();

    fn init(&mut self) -> Result<(), Error> {
        let param = self.cur_param();
        let cost = self.apply(&param)?;
        let grad = self.gradient(&param)?;
        self.p = grad.scale(-1.0);
        self.set_cur_cost(cost);
        self.set_cur_grad(grad);
        Ok(())
    }

    /// Perform one iteration of SA algorithm
    fn next_iter(&mut self) -> Result<ArgminIterationData<Self::Parameters>, Error> {
        // reset line search
        self.linesearch.base_reset();

        let xk = self.cur_param();
        let grad = self.cur_grad();
        let pk = self.p.clone();
        let cur_cost = self.cur_cost();
        self.linesearch.set_initial_parameter(xk);
        self.linesearch.set_search_direction(pk);
        self.linesearch.set_initial_gradient(grad.clone());
        self.linesearch.set_initial_cost(cur_cost);

        self.linesearch.run_fast()?;

        let xk1 = self.linesearch.result().param;

        let new_grad = self.gradient(&xk1)?;
        let new_grad_norm = new_grad.dot(new_grad.clone());

        // store this new dot product somewhere and reuse it in next iteration
        self.beta = new_grad_norm / grad.dot(grad.clone());

        self.p = new_grad.scale(-1.0).add(self.p.scale(self.beta));

        let mut out = ArgminIterationData::new(xk1, new_grad_norm);
        out.add_kv(make_kv!(
                "alpha" => self.alpha;
                "beta" => self.beta;
            ));
        Ok(out)
    }
}
