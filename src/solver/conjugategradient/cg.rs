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
    /// `cost_function`: cost function
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
        + ArgminSub<P, P>
        + ArgminDot<P, f64>
        + ArgminScaledAdd<P, f64, P>
        + ArgminAdd<P, P>
        + ArgminMul<f64, P>
        + ArgminDot<P, f64>,
{
    fn init(
        &mut self,
        op: &mut OpWrapper<O>,
        state: IterState<P, O::Hessian>,
    ) -> Result<Option<ArgminIterData<O>>, Error> {
        let init_param = state.cur_param;
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
        state: IterState<P, O::Hessian>,
    ) -> Result<ArgminIterData<O>, Error> {
        self.p_prev = self.p.clone();
        let apk = op.apply(&self.p)?;
        self.alpha = self.rtr / self.p.dot(&apk);
        let new_param = state.cur_param.scaled_add(&self.alpha, &self.p);
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
