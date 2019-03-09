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
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::default::Default;

/// The nonlinear conjugate gradient is a generalization of the conjugate gradient method for
/// nonlinear optimization problems.
///
/// # Example
///
/// ```
/// # extern crate argmin;
/// use argmin::prelude::*;
/// use argmin::solver::conjugategradient::{NonlinearConjugateGradient, PolakRibiere};
/// use argmin::solver::linesearch::MoreThuenteLineSearch;
/// use argmin::testfunctions::{rosenbrock_2d, rosenbrock_2d_derivative};
/// # use serde::{Deserialize, Serialize};
///
/// # #[derive(Clone, Default, Serialize, Deserialize)]
/// # struct MyProblem {}
/// #
/// # impl ArgminOp for MyProblem {
/// #     type Param = Vec<f64>;
/// #     type Output = f64;
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
/// // set up line search
/// let linesearch = MoreThuenteLineSearch::new(operator.clone());
/// // set up beta update method
/// let beta_method = PolakRibiere::new();
///
/// // Set up nonlinear conjugate gradient method
/// let mut solver = NonlinearConjugateGradient::new(operator,
///                                                  init_param,
///                                                  linesearch,
///                                                  beta_method)?;
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
#[derive(Serialize, Deserialize)]
pub struct NonlinearConjugateGradient<P, L, B> {
    /// p
    p: P,
    /// beta
    beta: f64,
    /// line search
    linesearch: L,
    /// beta update method
    beta_method: B,
    /// Number of iterations after which a restart is performed
    restart_iter: u64,
    /// Restart based on orthogonality
    restart_orthogonality: Option<f64>,
}

impl<P, L, B> NonlinearConjugateGradient<P, L, B>
where
    P: Default,
    // where
    //     L: ArgminLineSearch<Param = O::Param, Output = f64, Hessian = O::Hessian>,
    //     B: ArgminNLCGBetaUpdate<O::Param>,
{
    /// Constructor (Polak Ribiere Conjugate Gradient (PR-CG))
    pub fn new(linesearch: L, beta_method: B) -> Result<Self, Error> {
        Ok(NonlinearConjugateGradient {
            p: P::default(),
            beta: std::f64::NAN,
            linesearch: linesearch,
            beta_method: beta_method,
            restart_iter: std::u64::MAX,
            restart_orthogonality: None,
        })
    }

    /// Specifiy the number of iterations after which a restart should be performed
    /// This allows the algorithm to "forget" previous information which may not be helpful
    /// anymore.
    pub fn restart_iters(mut self, iters: u64) -> Self {
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
    pub fn restart_orthogonality(mut self, v: f64) -> Self {
        self.restart_orthogonality = Some(v);
        self
    }
}

impl<O, P, L, B> Solver<O> for NonlinearConjugateGradient<P, L, B>
where
    O: ArgminOp<Param = P, Output = f64>,
    P: Clone
        + Default
        + Serialize
        + DeserializeOwned
        + ArgminSub<P, P>
        + ArgminDot<P, f64>
        + ArgminScaledAdd<P, f64, P>
        + ArgminAdd<P, P>
        + ArgminMul<f64, P>
        + ArgminDot<P, f64>
        + ArgminNorm<f64>,
    O::Hessian: Default,
    L: Clone + ArgminLineSearch<P> + Solver<OpWrapper<O>>,
    B: ArgminNLCGBetaUpdate<P>,
{
    fn init(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &IterState<O>,
    ) -> Result<Option<ArgminIterData<O>>, Error> {
        let param = state.get_param();
        let cost = op.apply(&param)?;
        let grad = op.gradient(&param)?;
        self.p = grad.mul(&(-1.0));
        Ok(Some(
            ArgminIterData::new().param(param).cost(cost).grad(grad),
        ))
    }

    /// Perform one iteration of SA algorithm
    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &IterState<O>,
    ) -> Result<ArgminIterData<O>, Error> {
        let xk = state.get_param();
        let grad = if let Some(grad) = state.get_grad() {
            grad
        } else {
            op.gradient(&xk)?
        };
        let cur_cost = state.get_cost();

        // Linesearch
        self.linesearch.set_search_direction(self.p.clone());

        // Run solver
        let linesearch_result = Executor::new(op.clone(), self.linesearch.clone(), xk)
            .grad(grad.clone())
            .cost(cur_cost)
            .run_fast()?;

        // takes care of the counts of function evaluations
        op.consume_op(linesearch_result.operator);

        let xk1 = linesearch_result.param;

        // Update of beta
        let new_grad = op.gradient(&xk1)?;

        let restart_orthogonality = match self.restart_orthogonality {
            Some(v) => new_grad.dot(&grad).abs() / new_grad.norm().powi(2) >= v,
            None => false,
        };

        let restart_iter: bool =
            (state.get_iter() % self.restart_iter == 0) && state.get_iter() != 0;

        if restart_iter || restart_orthogonality {
            self.beta = 0.0;
        } else {
            self.beta = self.beta_method.update(&grad, &new_grad, &self.p);
        }

        // Update of p
        self.p = new_grad.mul(&(-1.0)).add(&self.p.mul(&self.beta));

        // Housekeeping
        let cost = op.apply(&xk1)?;

        Ok(ArgminIterData::new()
            .param(xk1)
            .cost(cost)
            .grad(new_grad)
            .kv(make_kv!("beta" => self.beta;
             "restart_iter" => restart_iter;
             "restart_orthogonality" => restart_orthogonality;
            )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::send_sync_test;
    use crate::solver::conjugategradient::beta::PolakRibiere;
    use crate::solver::linesearch::MoreThuenteLineSearch;
    use crate::MinimalNoOperator;

    send_sync_test!(
        nonlinear_cg,
        NonlinearConjugateGradient<
            MinimalNoOperator,
            MoreThuenteLineSearch<MinimalNoOperator>,
            PolakRibiere,
        >
    );
}
