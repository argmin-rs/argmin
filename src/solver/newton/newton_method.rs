// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Newton's method
//!
//! # References:
//!
//! [0] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
//! Springer. ISBN 0-387-30303-0.

use prelude::*;
use std;
use std::default::Default;

/// Newton's Method
#[derive(ArgminSolver)]
pub struct Newton<'a, T, H>
where
    T: 'a + Clone + Default + ArgminScaledSub<T, f64>,
    H: 'a + Clone + Default + ArgminInv<H> + ArgminDot<T, T>,
{
    /// gamma
    gamma: f64,
    /// Base stuff
    base: ArgminBase<'a, T, f64, H>,
}

impl<'a, T, H> Newton<'a, T, H>
where
    T: 'a + Clone + Default + ArgminScaledSub<T, f64>,
    H: 'a + Clone + Default + ArgminInv<H> + ArgminDot<T, T>,
{
    /// Constructor
    pub fn new(
        cost_function: &'a ArgminOperator<Parameters = T, OperatorOutput = f64, Hessian = H>,
        init_param: T,
    ) -> Self {
        Newton {
            gamma: 1.0,
            base: ArgminBase::new(cost_function, init_param),
        }
    }

    /// set gamma
    pub fn set_gamma(&mut self, gamma: f64) -> Result<&mut Self, Error> {
        if gamma <= 0.0 || gamma > 1.0 {
            return Err(ArgminError::InvalidParameter {
                text: "Newton: gamma must be in  (0, 1].".to_string(),
            }
            .into());
        }
        self.gamma = gamma;
        Ok(self)
    }
}

impl<'a, T, H> ArgminNextIter for Newton<'a, T, H>
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
        let hessian = self.hessian(&param)?;
        let new_param = param.scaled_sub(self.gamma, hessian.ainv()?.dot(grad));
        let out = ArgminIterationData::new(new_param, 0.0);
        Ok(out)
    }
}
