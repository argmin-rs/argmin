// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Important TODO: Find out which line search should be the default choice. Also try to replicate
//! CG_DESCENT.
//!
//! # References:
//!
//! [0] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
//! Springer. ISBN 0-387-30303-0.

use crate::prelude::*;
use crate::solver::conjugategradient::{
    FletcherReeves, HestenesStiefel, PolakRibiere, PolakRibierePlus,
};
use crate::solver::linesearch::HagerZhangLineSearch;
use std;
use std::default::Default;

/// The nonlinear conjugate gradient is a generalization of the conjugate gradient method for
/// nonlinear optimization problems.
///
/// # Example
///
/// ```
/// # extern crate argmin;
/// use argmin::prelude::*;
/// use argmin::solver::conjugategradient::NonlinearConjugateGradient;
/// use argmin::testfunctions::{rosenbrock_2d, rosenbrock_2d_derivative};
///
/// # #[derive(Clone)]
/// # struct MyProblem {}
/// #
/// # impl ArgminOperator for MyProblem {
/// #     type Parameters = Vec<f64>;
/// #     type OperatorOutput = f64;
/// #     type Hessian = ();
/// #
/// #     fn apply(&self, p: &Vec<f64>) -> Result<f64, Error> {
/// #         Ok(rosenbrock_2d(p, 1.0, 100.0))
/// #     }
/// #
/// #     fn gradient(&self, p: &Vec<f64>) -> Result<Vec<f64>, Error> {
/// #         Ok(rosenbrock_2d_derivative(p, 1.0, 100.0))
/// #     }
/// # }
/// #
/// # fn run() -> Result<(), Error> {
/// // Set up cost function
/// let operator = MyProblem {};
///
/// // define inital parameter vector
/// let init_param: Vec<f64> = vec![1.2, 1.2];
///
/// // Set up nonlinear conjugate gradient method
/// let mut solver = NonlinearConjugateGradient::new_pr(&operator, init_param)?;
///
/// // Set maximum number of iterations
/// solver.set_max_iters(20);
///
/// // Set target cost function value
/// solver.set_target_cost(0.0);
///
/// // Set the number of iterations when a restart should be performed
/// // This allows the algorithm to "forget" previous information which may not be helpful anymore.
/// solver.set_restart_iters(10);
///
/// // Set the value for the orthogonality measure.
/// // Setting this parameter leads to a restart of the algorithm (setting beta = 0) after two
/// // consecutive search directions are not orthogonal anymore. In other words, if this condition
/// // is met:
/// //
/// // `|\nabla f_k^T * \nabla f_{k-1}| / | \nabla f_k ||^2 >= v`
/// //
/// // A typical value for `v` is 0.1.
/// solver.set_restart_orthogonality(0.1);
///
/// // Attach a logger
/// solver.add_logger(ArgminSlogLogger::term());
///
/// // Run solver
/// solver.run()?;
///
/// // Wait a second (lets the logger flush everything before printing to screen again)
/// std::thread::sleep(std::time::Duration::from_secs(1));
///
/// // Print result
/// println!("{:?}", solver.result());
/// #     Ok(())
/// # }
/// #
/// # fn main() {
/// #     if let Err(ref e) = run() {
/// #         println!("{} {}", e.as_fail(), e.backtrace());
/// #     }
/// # }
/// ```
///
/// # References:
///
/// [0] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(ArgminSolver)]
pub struct NonlinearConjugateGradient<'a, T>
where
    T: 'a
        + Clone
        + Default
        + ArgminSub<T>
        + ArgminAdd<T, T>
        + ArgminScale<f64>
        + ArgminNorm<f64>
        + ArgminDot<T, f64>
        + ArgminScaledAdd<T, f64>
        + ArgminScaledSub<T, f64>,
{
    /// p
    p: T,
    /// beta
    beta: f64,
    /// line search
    linesearch: Box<ArgminLineSearch<Parameters = T, OperatorOutput = f64, Hessian = ()> + 'a>,
    /// beta update method
    beta_method: Box<ArgminNLCGBetaUpdate<T> + 'a>,
    /// Number of iterations after which a restart is performed
    restart_iter: u64,
    /// Restart based on orthogonality
    restart_orthogonality: Option<f64>,
    /// base
    base: ArgminBase<'a, T, f64, ()>,
}

impl<'a, T> NonlinearConjugateGradient<'a, T>
where
    T: 'a
        + Clone
        + Default
        + ArgminSub<T>
        + ArgminAdd<T, T>
        + ArgminScale<f64>
        + ArgminNorm<f64>
        + ArgminDot<T, f64>
        + ArgminScaledAdd<T, f64>
        + ArgminScaledSub<T, f64>,
{
    /// Constructor (Polak Ribiere Conjugate Gradient (PR-CG))
    pub fn new(
        operator: &'a ArgminOperator<Parameters = T, OperatorOutput = f64, Hessian = ()>,
        init_param: T,
    ) -> Result<Self, Error> {
        let linesearch: Box<dyn ArgminLineSearch<Parameters = _, OperatorOutput = _, Hessian = _>> =
            Box::new(HagerZhangLineSearch::new(operator));
        let beta_method = PolakRibiere::new();
        Ok(NonlinearConjugateGradient {
            p: T::default(),
            beta: std::f64::NAN,
            linesearch,
            beta_method: Box::new(beta_method),
            restart_iter: std::u64::MAX,
            restart_orthogonality: None,
            base: ArgminBase::new(operator, init_param),
        })
    }

    /// New PolakRibiere CG (PR-CG)
    pub fn new_pr(
        operator: &'a ArgminOperator<Parameters = T, OperatorOutput = f64, Hessian = ()>,
        init_param: T,
    ) -> Result<Self, Error> {
        Self::new(operator, init_param)
    }

    /// New PolakRibierePlus CG (PR+-CG)
    pub fn new_prplus(
        operator: &'a ArgminOperator<Parameters = T, OperatorOutput = f64, Hessian = ()>,
        init_param: T,
    ) -> Result<Self, Error> {
        let mut s = Self::new(operator, init_param)?;
        let beta_method = PolakRibierePlus::new();
        s.set_beta_update(Box::new(beta_method));
        Ok(s)
    }

    /// New FletcherReeves CG (FR-CG)
    pub fn new_fr(
        operator: &'a ArgminOperator<Parameters = T, OperatorOutput = f64, Hessian = ()>,
        init_param: T,
    ) -> Result<Self, Error> {
        let mut s = Self::new(operator, init_param)?;
        let beta_method = FletcherReeves::new();
        s.set_beta_update(Box::new(beta_method));
        Ok(s)
    }

    /// New HestenesStiefel CG (HS-CG)
    pub fn new_hs(
        operator: &'a ArgminOperator<Parameters = T, OperatorOutput = f64, Hessian = ()>,
        init_param: T,
    ) -> Result<Self, Error> {
        let mut s = Self::new(operator, init_param)?;
        let beta_method = HestenesStiefel::new();
        s.set_beta_update(Box::new(beta_method));
        Ok(s)
    }

    /// Specify line search method
    pub fn set_linesearch(
        &mut self,
        linesearch: Box<ArgminLineSearch<Parameters = T, OperatorOutput = f64, Hessian = ()> + 'a>,
    ) -> &mut Self {
        self.linesearch = linesearch;
        self
    }

    /// Specify beta update method
    pub fn set_beta_update(&mut self, beta_method: Box<ArgminNLCGBetaUpdate<T> + 'a>) -> &mut Self {
        self.beta_method = beta_method;
        self
    }

    /// Specifiy the number of iterations after which a restart should be performed
    /// This allows the algorithm to "forget" previous information which may not be helpful
    /// anymore.
    pub fn set_restart_iters(&mut self, iters: u64) -> &mut Self {
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
    pub fn set_restart_orthogonality(&mut self, v: f64) -> &mut Self {
        self.restart_orthogonality = Some(v);
        self
    }
}

impl<'a, T> ArgminNextIter for NonlinearConjugateGradient<'a, T>
where
    T: 'a
        + Clone
        + Default
        + ArgminSub<T>
        + ArgminAdd<T, T>
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

        // Linesearch
        self.linesearch.set_initial_parameter(xk);
        self.linesearch.set_search_direction(pk.clone());
        self.linesearch.set_initial_gradient(grad.clone());
        self.linesearch.set_initial_cost(cur_cost);

        self.linesearch.run_fast()?;

        let xk1 = self.linesearch.result().param;

        // Update of beta
        let new_grad = self.gradient(&xk1)?;

        let restart_orthogonality = match self.restart_orthogonality {
            Some(v) => new_grad.dot(&grad).abs() / new_grad.norm().powi(2) >= v,
            None => false,
        };

        let restart_iter: bool = (self.cur_iter() % self.restart_iter == 0) && self.cur_iter() != 0;

        if restart_iter || restart_orthogonality {
            self.beta = 0.0;
        } else {
            self.beta = self.beta_method.update(&grad, &new_grad, &pk);
        }

        // Update of p
        self.p = new_grad.scale(-1.0).add(&self.p.scale(self.beta));

        // Housekeeping
        self.set_cur_param(xk1.clone());
        self.set_cur_grad(new_grad);
        let cost = self.apply(&xk1)?;
        self.set_cur_cost(cost);

        let mut out = ArgminIterationData::new(xk1, cost);
        out.add_kv(make_kv!(
            "beta" => self.beta;
            "restart_iter" => restart_iter;
            "restart_orthogonality" => restart_orthogonality;
        ));
        Ok(out)
    }
}
