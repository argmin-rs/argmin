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
use std;
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
///
/// #[derive(Clone)]
/// struct MyProblem {}
///
/// impl ArgminOperator for MyProblem {
///     type Parameters = Vec<f64>;
///     type OperatorOutput = Vec<f64>;
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
/// let mut solver = ConjugateGradient::new(&operator, b, init_param)?;
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
#[derive(ArgminSolver)]
pub struct ConjugateGradient<'a, T>
where
    T: Clone
        + Default
        + ArgminSub<T, T>
        + ArgminAdd<T, T>
        + ArgminScale<f64>
        + ArgminDot<T, f64>
        + ArgminScaledAdd<T, f64>
        + ArgminScaledSub<T, f64>,
{
    /// b (right hand side)
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
        + ArgminSub<T, T>
        + ArgminAdd<T, T>
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
    ///
    /// `b`: right hand side of `A * x = b`
    ///
    /// `init_param`: Initial parameter vector
    pub fn new(
        operator: &'a ArgminOperator<Parameters = T, OperatorOutput = T, Hessian = ()>,
        b: T,
        init_param: T,
    ) -> Result<Self, Error> {
        Ok(ConjugateGradient {
            b,
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
        + ArgminSub<T, T>
        + ArgminAdd<T, T>
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
        let r0 = self.b.sub(&ap).scale(-1.0);
        self.r = r0.clone();
        self.p = r0.scale(-1.0);
        self.rtr = self.r.dot(&self.r);
        Ok(())
    }

    /// Perform one iteration of SA algorithm
    fn next_iter(&mut self) -> Result<ArgminIterationData<Self::Parameters>, Error> {
        // Still way too much cloning going on here
        self.p_prev = self.p.clone();
        let p = self.p.clone();
        let apk = self.apply(&p)?;
        self.alpha = self.rtr / self.p.dot(&apk);
        let new_param = self.cur_param().scaled_add(self.alpha, &p);
        self.r = self.r.scaled_add(self.alpha, &apk);
        let rtr_n = self.r.dot(&self.r);
        self.beta = rtr_n / self.rtr;
        self.rtr = rtr_n;
        self.p = self.r.scale(-1.0).scaled_add(self.beta, &p);
        let norm = self.r.dot(&self.r);

        let mut out = ArgminIterationData::new(new_param, norm.sqrt());
        out.add_kv(make_kv!(
            "alpha" => self.alpha;
            "beta" => self.beta;
        ));
        Ok(out)
    }
}
