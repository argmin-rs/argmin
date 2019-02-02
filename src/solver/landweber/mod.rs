// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Landweber iteration
//!
//! [Landweber](struct.Landweber.html)
//!
//! # References
//!
//! [0] Landweber, L. (1951): An iteration formula for Fredholm integral equations of the first
//! kind. Amer. J. Math. 73, 615–624
//! [1] https://en.wikipedia.org/wiki/Landweber_iteration

use crate::prelude::*;
use std;
use std::default::Default;

/// The Landweber iteration is a solver for ill-posed linear inverse problems.
///
/// In iteration `k`, the new parameter vector `x_{k+1}` is calculated from the previous parameter
/// vector `x_k` and the gradient at `x_k` according to the following update rule:
///
/// `x_{k+1} = x_k - omega * \nabla f(x_k)`
///
/// # Example
///
/// ```
/// # extern crate argmin;
/// use argmin::prelude::*;
/// use argmin::solver::landweber::Landweber;
/// # use argmin::testfunctions::{rosenbrock_2d, rosenbrock_2d_derivative};
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
/// let operator = MyProblem {};
/// let init_param: Vec<f64> = vec![1.2, 1.2];
/// let omega = 0.001;
///
/// let mut solver = Landweber::new(&operator, omega, init_param)?;
/// solver.set_max_iters(100);
/// solver.add_logger(ArgminSlogLogger::term());
/// solver.run()?;
///
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
/// # References
///
/// [0] Landweber, L. (1951): An iteration formula for Fredholm integral equations of the first
/// kind. Amer. J. Math. 73, 615–624
/// [1] https://en.wikipedia.org/wiki/Landweber_iteration
#[derive(ArgminSolver)]
pub struct Landweber<'a, T>
where
    T: 'a + Clone + Default + ArgminScaledSub<T, f64, T>,
{
    /// omgea
    omega: f64,
    /// Base stuff
    base: ArgminBase<'a, T, f64, ()>,
}

impl<'a, T> Landweber<'a, T>
where
    T: 'a + Clone + Default + ArgminScaledSub<T, f64, T>,
{
    /// Constructor
    pub fn new(
        cost_function: &'a ArgminOperator<Parameters = T, OperatorOutput = f64, Hessian = ()>,
        omega: f64,
        init_param: T,
    ) -> Result<Self, Error> {
        Ok(Landweber {
            omega,
            base: ArgminBase::new(cost_function, init_param),
        })
    }
}

impl<'a, T> ArgminNextIter for Landweber<'a, T>
where
    T: 'a + Clone + Default + ArgminScaledSub<T, f64, T>,
{
    type Parameters = T;
    type OperatorOutput = f64;
    type Hessian = ();

    fn next_iter(&mut self) -> Result<ArgminIterationData<Self::Parameters>, Error> {
        let param = self.cur_param();
        let grad = self.gradient(&param)?;
        let new_param = param.scaled_sub(&self.omega, &grad);
        let out = ArgminIterationData::new(new_param, 0.0);
        Ok(out)
    }
}
