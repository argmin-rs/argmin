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
#[derive(Serialize, Deserialize)]
pub struct Landweber {
    /// omgea
    omega: f64,
}

impl Landweber {
    /// Constructor
    pub fn new(omega: f64) -> Result<Self, Error> {
        Ok(Landweber { omega })
    }
}

impl<O> Solver<O> for Landweber
where
    O: ArgminOp,
    O::Param: ArgminScaledSub<O::Param, f64, O::Param>,
{
    fn next_iter<'a>(
        &mut self,
        op: &mut OpWrapper<'a, O>,
        state: IterState<O::Param, O::Hessian>,
    ) -> Result<ArgminIterData<O::Param>, Error> {
        let grad = op.gradient(&state.cur_param)?;
        let new_param = state.cur_param.scaled_sub(&self.omega, &grad);
        let out = ArgminIterData::new(new_param, 0.0);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::send_sync_test;

    send_sync_test!(landweber, Landweber);
}
