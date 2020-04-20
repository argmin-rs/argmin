// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # `ArgminResult`
//!
//! Returned by a solver and consists of the used operator and the last `IterState` of the solver.
//! Both can be accessed by the methods `operator()` and `state()`.
//!
//! The reference to the struct returned by `state()` allows one to for instance access the final
//! parameter vector or the final cost function value.
//!
//! ## Examples:
//!
//! ```
//! # #![allow(unused_imports)]
//! # extern crate argmin;
//! # extern crate argmin_testfunctions;
//! # use argmin::prelude::*;
//! # use argmin::solver::gradientdescent::SteepestDescent;
//! # use argmin::solver::linesearch::MoreThuenteLineSearch;
//! # use argmin_testfunctions::{rosenbrock_2d, rosenbrock_2d_derivative};
//! # use serde::{Deserialize, Serialize};
//! #
//! # #[derive(Clone, Default, Serialize, Deserialize)]
//! # struct Rosenbrock {
//! #     a: f64,
//! #     b: f64,
//! # }
//! #
//! # impl ArgminOp for Rosenbrock {
//! #     type Param = Vec<f64>;
//! #     type Output = f64;
//! #     type Hessian = ();
//! #     type Jacobian = ();
//! #     type Float = f64;
//! #
//! #     fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
//! #         Ok(rosenbrock_2d(p, self.a, self.b))
//! #     }
//! #
//! #     fn gradient(&self, p: &Self::Param) -> Result<Self::Param, Error> {
//! #         Ok(rosenbrock_2d_derivative(p, self.a, self.b))
//! #     }
//! # }
//! #
//! # fn run() -> Result<(), Error> {
//! #     // Define cost function (must implement `ArgminOperator`)
//! #     let cost = Rosenbrock { a: 1.0, b: 100.0 };
//! #     // Define initial parameter vector
//! #     let init_param: Vec<f64> = vec![-1.2, 1.0];
//! #     // Set up line search
//! #     let linesearch = MoreThuenteLineSearch::new();
//! #     // Set up solver
//! #     let solver = SteepestDescent::new(linesearch);
//! #     // Run solver
//! #     let result = Executor::new(cost, solver, init_param)
//! #         // Set maximum iterations to 10
//! #         .max_iters(1)
//! #         // run the solver on the defined problem
//! #         .run()?;
//! // Get best parameter vector
//! let best_parameter = result.state().get_best_param();
//!
//! // Get best cost function value
//! let best_cost = result.state().get_best_cost();
//!
//! // Get the number of iterations
//! let num_iters = result.state().get_iter();
//! #     Ok(())
//! # }
//! #
//! # fn main() {
//! #     if let Err(ref e) = run() {
//! #         println!("{}", e);
//! #         std::process::exit(1);
//! #     }
//! # }
//! ```
//!
//! More details can be found in the `IterState` documentation.

use crate::prelude::*;
use std::cmp::Ordering;

/// Final struct returned by the `run` method of `Executor`.
#[derive(Clone)]
pub struct ArgminResult<O: ArgminOp> {
    /// operator
    pub operator: O,
    /// iteration state
    pub state: IterState<O>,
}

impl<O: ArgminOp> ArgminResult<O> {
    /// Constructor
    pub fn new(operator: O, state: IterState<O>) -> Self {
        ArgminResult { operator, state }
    }

    /// Return handle to operator
    pub fn operator(&self) -> &O {
        &self.operator
    }

    /// Return handle to state
    pub fn state(&self) -> &IterState<O> {
        &self.state
    }
}

impl<O> std::fmt::Display for ArgminResult<O>
where
    O: ArgminOp,
    O::Param: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(f, "ArgminResult:")?;
        writeln!(f, "    param (best):  {:?}", self.state.get_best_param())?;
        writeln!(f, "    cost (best):   {}", self.state.get_best_cost())?;
        writeln!(f, "    iters (best):  {}", self.state.get_last_best_iter())?;
        writeln!(f, "    iters (total): {}", self.state.get_iter())?;
        writeln!(
            f,
            "    termination: {}",
            self.state.get_termination_reason()
        )?;
        writeln!(f, "    time:        {:?}", self.state.get_time())?;
        Ok(())
    }
}

impl<O: ArgminOp> PartialEq for ArgminResult<O> {
    fn eq(&self, other: &ArgminResult<O>) -> bool {
        (self.state.get_cost() - other.state.get_cost()).abs() < O::Float::epsilon()
    }
}

impl<O: ArgminOp> Eq for ArgminResult<O> {}

impl<O: ArgminOp> Ord for ArgminResult<O> {
    fn cmp(&self, other: &ArgminResult<O>) -> Ordering {
        let t = self.state.get_cost() - other.state.get_cost();
        if t.abs() < O::Float::epsilon() {
            Ordering::Equal
        } else if t > O::Float::from_f64(0.0).unwrap() {
            Ordering::Greater
        } else {
            Ordering::Less
        }
    }
}

impl<O: ArgminOp> PartialOrd for ArgminResult<O> {
    fn partial_cmp(&self, other: &ArgminResult<O>) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::MinimalNoOperator;

    send_sync_test!(argmin_result, ArgminResult<MinimalNoOperator>);
}
