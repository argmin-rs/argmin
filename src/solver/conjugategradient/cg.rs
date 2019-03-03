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
use serde::{Deserialize, Serialize};
use std::default::Default;

/// The conjugate gradient method is a solver for systems of linear equations with a symmetric and
/// positive-definite matrix.
///
/// # Example
///
/// ```
/// extern crate argmin;
/// use argmin::prelude::*;
/// use argmin::solver::conjugategradient::ConjugateGradient;
/// use serde::{Deserialize, Serialize};
///
/// #[derive(Clone, Default, Serialize, Deserialize)]
/// struct MyProblem {}
///
/// impl ArgminOp for MyProblem {
///     type Param = Vec<f64>;
///     type Output = Vec<f64>;
///     type Hessian = ();
///
///     fn apply(&self, p: &Vec<f64>) -> Result<Vec<f64>, Error> {
///         Ok(vec![4.0 * p[0] + 1.0 * p[1], 1.0 * p[0] + 3.0 * p[1]])
///     }
/// }
///
/// # fn run() -> Result<(), Error> {
/// // Define inital parameter vector
/// let init_param: Vec<f64> = vec![2.0, 1.0];
///
/// // Define the right hand side `b` of `A * x = b`
/// let b = vec![1.0, 2.0];
///
/// // Set up operator
/// let operator = MyProblem {};
///
/// // Set up the solver
/// let mut solver = ConjugateGradient::new(operator, b, init_param)?;
///
/// // Set maximum number of iterations
/// solver.set_max_iters(2);
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
#[derive(ArgminSolver, Clone, Serialize, Deserialize)]
pub struct ConjugateGradient<O>
where
    O: ArgminOp<Output = <O as ArgminOp>::Param>,
    O::Param: ArgminSub<O::Param, O::Param>
        + ArgminDot<O::Param, f64>
        + ArgminScaledAdd<O::Param, f64, O::Param>
        + ArgminAdd<O::Param, O::Param>
        + ArgminMul<f64, O::Param>
        + ArgminDot<O::Param, f64>,
{
    /// b (right hand side)
    b: O::Param,
    /// residual
    r: O::Param,
    /// p
    p: O::Param,
    /// previous p
    p_prev: O::Param,
    /// r^T * r
    rtr: f64,
    /// alpha
    alpha: f64,
    /// beta
    beta: f64,
    /// base
    base: ArgminBase<O>,
}

impl<O> ConjugateGradient<O>
where
    O: ArgminOp<Output = <O as ArgminOp>::Param>,
    O::Param: ArgminSub<O::Param, O::Param>
        + ArgminDot<O::Param, f64>
        + ArgminScaledAdd<O::Param, f64, O::Param>
        + ArgminAdd<O::Param, O::Param>
        + ArgminMul<f64, O::Param>
        + ArgminDot<O::Param, f64>,
{
    /// Constructor
    ///
    /// Parameters:
    ///
    /// `cost_function`: cost function
    ///
    /// `b`: right hand side of `A * x = b`
    ///
    /// `init_param`: Initial parameter vector
    pub fn new(operator: O, b: O::Param, init_param: O::Param) -> Result<Self, Error> {
        Ok(ConjugateGradient {
            b,
            r: O::Param::default(),
            p: O::Param::default(),
            p_prev: O::Param::default(),
            rtr: std::f64::NAN,
            alpha: std::f64::NAN,
            beta: std::f64::NAN,
            base: ArgminBase::new(operator, init_param),
        })
    }

    /// Return the current search direction (This is needed by NewtonCG for instance)
    pub fn p(&self) -> O::Param {
        self.p.clone()
    }

    /// Return the previous search direction (This is needed by NewtonCG for instance)
    pub fn p_prev(&self) -> O::Param {
        self.p_prev.clone()
    }

    /// Return the current residual (This is needed by NewtonCG for instance)
    pub fn residual(&self) -> O::Param {
        self.r.clone()
    }
}

impl<O> ArgminIter for ConjugateGradient<O>
where
    O: ArgminOp<Output = <O as ArgminOp>::Param>,
    O::Param: ArgminSub<O::Param, O::Param>
        + ArgminDot<O::Param, f64>
        + ArgminScaledAdd<O::Param, f64, O::Param>
        + ArgminAdd<O::Param, O::Param>
        + ArgminMul<f64, O::Param>
        + ArgminDot<O::Param, f64>,
{
    type Param = O::Param;
    type Output = O::Output;
    type Hessian = O::Hessian;

    fn init(&mut self) -> Result<(), Error> {
        let init_param = self.cur_param();
        let ap = self.apply(&init_param)?;
        let r0 = self.b.sub(&ap).mul(&(-1.0));
        self.r = r0.clone();
        self.p = r0.mul(&(-1.0));
        self.rtr = self.r.dot(&self.r);
        Ok(())
    }

    /// Perform one iteration of SA algorithm
    fn next_iter(&mut self) -> Result<ArgminIterData<Self::Param>, Error> {
        // Still way too much cloning going on here
        self.p_prev = self.p.clone();
        let p = self.p.clone();
        let apk = self.apply(&p)?;
        self.alpha = self.rtr / self.p.dot(&apk);
        let new_param = self.cur_param().scaled_add(&self.alpha, &p);
        self.r = self.r.scaled_add(&self.alpha, &apk);
        let rtr_n = self.r.dot(&self.r);
        self.beta = rtr_n / self.rtr;
        self.rtr = rtr_n;
        self.p = self.r.mul(&(-1.0)).scaled_add(&self.beta, &p);
        let norm = self.r.dot(&self.r);

        let mut out = ArgminIterData::new(new_param, norm.sqrt());
        out.add_kv(make_kv!(
            "alpha" => self.alpha;
            "beta" => self.beta;
        ));
        Ok(out)
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
