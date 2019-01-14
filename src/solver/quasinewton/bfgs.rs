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

/// Text
///
/// # Example
///
///
/// # References:
///
/// [0] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
/// Springer. ISBN 0-387-30303-0.
#[derive(ArgminSolver)]
pub struct BFGS<'a, T, H>
where
    T: 'a + Clone + Default + ArgminScaledSub<T, f64>,
    H: 'a + Clone + Default + ArgminInv<H> + ArgminDot<T, T>,
{
    /// Base stuff
    base: ArgminBase<'a, T, f64, H>,
}

impl<'a, T, H> BFGS<'a, T, H>
where
    T: 'a + Clone + Default + ArgminScaledSub<T, f64>,
    H: 'a + Clone + Default + ArgminInv<H> + ArgminDot<T, T>,
{
    /// Constructor
    pub fn new(
        cost_function: &'a ArgminOperator<Parameters = T, OperatorOutput = f64, Hessian = H>,
        init_param: T,
        init_inverse_hessian: H,
    ) -> Self {
        let mut base = ArgminBase::new(cost_function, init_param);
        base.set_cur_hessian(init_inverse_hessian);
        BFGS { base }
    }
}

impl<'a, T, H> ArgminNextIter for BFGS<'a, T, H>
where
    T: 'a + Clone + Default + ArgminScaledSub<T, f64>,
    H: 'a + Clone + Default + ArgminInv<H> + ArgminDot<T, T>,
{
    type Parameters = T;
    type OperatorOutput = f64;
    type Hessian = H;

    fn next_iter(&mut self) -> Result<ArgminIterationData<Self::Parameters>, Error> {
        let param = self.cur_param();
        let grad = self.gradient(&param)?;
        let inv_hessian = self.cur_hessian();
        let p = inv_hessian.dot(&grad);
        // let new_param = param.scaled_sub(self.gamma, &hessian.ainv()?.dot(&grad));
        // let out = ArgminIterationData::new(new_param, 0.0);
        // Ok(out)
        unimplemented!()
    }
}
