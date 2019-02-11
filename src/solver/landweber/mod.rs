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
use serde::{Deserialize, Serialize};

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
/// let operator = MyProblem {};
/// let init_param: Vec<f64> = vec![1.2, 1.2];
/// let omega = 0.001;
///
/// let mut solver = Landweber::new(operator, omega, init_param)?;
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
#[derive(ArgminSolver, Serialize, Deserialize)]
pub struct Landweber<O>
where
    <O as ArgminOp>::Param: ArgminScaledSub<<O as ArgminOp>::Param, f64, <O as ArgminOp>::Param>,
    O: ArgminOp,
{
    /// omgea
    omega: f64,
    /// Base stuff
    base: ArgminBase<O>,
}

impl<O> Landweber<O>
where
    <O as ArgminOp>::Param: ArgminScaledSub<<O as ArgminOp>::Param, f64, <O as ArgminOp>::Param>,
    O: ArgminOp,
{
    /// Constructor
    pub fn new(
        cost_function: O,
        omega: f64,
        init_param: <O as ArgminOp>::Param,
    ) -> Result<Self, Error> {
        Ok(Landweber {
            omega,
            base: ArgminBase::new(cost_function, init_param),
        })
    }
}

impl<O> ArgminIter for Landweber<O>
where
    <O as ArgminOp>::Param: ArgminScaledSub<<O as ArgminOp>::Param, f64, <O as ArgminOp>::Param>,
    O: ArgminOp,
{
    type Param = <O as ArgminOp>::Param;
    type Output = <O as ArgminOp>::Output;
    type Hessian = <O as ArgminOp>::Hessian;

    fn next_iter(&mut self) -> Result<ArgminIterData<Self::Param>, Error> {
        let param = self.cur_param();
        let grad = self.gradient(&param)?;
        let new_param = param.scaled_sub(&self.omega, &grad);
        let out = ArgminIterData::new(new_param, 0.0);
        Ok(out)
    }
}
