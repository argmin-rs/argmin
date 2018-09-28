// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Hager-Zang line search algorithm
//!
//!
//! ## Reference
//!
//! William W. Hager and Hongchao Zhang. "A new conjugate gradient method with guaranteed descent
//! and an efficient line search." SIAM J. Optim. 16(1), 2006, 170-192.
//! DOI: https://doi.org/10.1137/030601880
//!
// //!
// //! # Example
// //!
// //! ```rust
// //! todo
// //! ```

use prelude::*;
use std;

/// More-Thuente Line Search
#[derive(ArgminSolver)]
pub struct HagerZhangLineSearch<T>
where
    T: std::default::Default
        + Clone
        + std::fmt::Debug
        + ArgminSub<T>
        + ArgminDot<T, f64>
        + ArgminScaledAdd<T, f64>
        + ArgminScaledSub<T, f64>,
{
    /// base
    base: ArgminBase<T, f64>,
}

impl<T> HagerZhangLineSearch<T>
where
    T: std::default::Default
        + Clone
        + std::fmt::Debug
        + ArgminSub<T>
        + ArgminDot<T, f64>
        + ArgminScaledAdd<T, f64>
        + ArgminScaledSub<T, f64>,
    HagerZhangLineSearch<T>: ArgminSolver<Parameters = T, OperatorOutput = f64>,
{
    /// Constructor
    ///
    /// Parameters:
    ///
    /// `operator`: operator
    pub fn new(operator: Box<ArgminOperator<Parameters = T, OperatorOutput = f64>>) -> Self {
        HagerZhangLineSearch {
            base: ArgminBase::new(operator, T::default()),
        }
    }

    /// set current gradient value
    pub fn set_cur_grad(&mut self, grad: T) -> &mut Self {
        self.base.set_cur_grad(grad);
        self
    }
}

impl<T> ArgminLineSearch for HagerZhangLineSearch<T>
where
    T: std::default::Default
        + Clone
        + std::fmt::Debug
        + ArgminSub<T>
        + ArgminDot<T, f64>
        + ArgminScaledAdd<T, f64>
        + ArgminScaledSub<T, f64>,
{
    /// Set search direction
    fn set_search_direction(&mut self, search_direction: T) {
        // self.search_direction_b = Some(search_direction);
        unimplemented!()
    }

    /// Set initial parameter
    fn set_initial_parameter(&mut self, param: T) {
        // self.init_param_b = Some(param.clone());
        // self.base.set_cur_param(param);
        unimplemented!()
    }

    /// Set initial cost function value
    fn set_initial_cost(&mut self, init_cost: f64) {
        // self.finit_b = Some(init_cost);
        unimplemented!()
    }

    /// Set initial gradient
    fn set_initial_gradient(&mut self, init_grad: T) {
        // self.init_grad_b = Some(init_grad);
        unimplemented!()
    }

    /// Calculate initial cost function value
    fn calc_initial_cost(&mut self) -> Result<(), Error> {
        // let tmp = self.base.cur_param();
        // self.finit_b = Some(self.apply(&tmp)?);
        // Ok(())
        unimplemented!()
    }

    /// Calculate initial cost function value
    fn calc_initial_gradient(&mut self) -> Result<(), Error> {
        // let tmp = self.base.cur_param();
        // self.init_grad_b = Some(self.gradient(&tmp)?);
        // Ok(())
        unimplemented!()
    }

    /// Set initial alpha value
    fn set_initial_alpha(&mut self, alpha: f64) -> Result<(), Error> {
        // if alpha <= 0.0 {
        //     return Err(ArgminError::InvalidParameter {
        //         parameter: "MoreThuenteLineSearch: Initial alpha must be > 0.".to_string(),
        //     }.into());
        // }
        // self.alpha = alpha;
        // Ok(())
        unimplemented!()
    }
}

impl<T> ArgminNextIter for HagerZhangLineSearch<T>
where
    T: std::default::Default
        + Clone
        + std::fmt::Debug
        + ArgminSub<T>
        + ArgminDot<T, f64>
        + ArgminScaledAdd<T, f64>
        + ArgminScaledSub<T, f64>,
{
    type Parameters = T;
    type OperatorOutput = f64;

    fn init(&mut self) -> Result<(), Error> {
        unimplemented!()
    }

    fn next_iter(&mut self) -> Result<ArgminIterationData<Self::Parameters>, Error> {
        // let out = ArgminIterationData::new(new_param, self.stp.fx);
        // Ok(out)
        unimplemented!()
    }
}
