// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.
//! # Landweber iteration
//!
//! # References
//!
//! [0] Landweber, L. (1951): An iteration formula for Fredholm integral equations of the first
//! kind. Amer. J. Math. 73, 615–624
//! [1] https://en.wikipedia.org/wiki/Landweber_iteration

use prelude::*;
use std;
use std::default::Default;

/// Landweber iteration
#[derive(ArgminSolver)]
pub struct Landweber<'a, T>
where
    T: 'a + Clone + Default + ArgminScaledSub<T, f64>,
{
    /// omgea
    omega: f64,
    /// Base stuff
    base: ArgminBase<'a, T, f64, ()>,
}

impl<'a, T> Landweber<'a, T>
where
    T: 'a + Clone + Default + ArgminScaledSub<T, f64>,
{
    /// Constructor
    pub fn new(
        cost_function: Box<ArgminOperator<Parameters = T, OperatorOutput = f64, Hessian = ()> + 'a>,
        omega: f64,
        init_param: T,
    ) -> Result<Self, Error> {
        Ok(Landweber {
            omega: omega,
            base: ArgminBase::new(cost_function, init_param),
        })
    }
}

impl<'a, T> ArgminNextIter for Landweber<'a, T>
where
    T: 'a + Clone + Default + ArgminScaledSub<T, f64>,
{
    type Parameters = T;
    type OperatorOutput = f64;
    type Hessian = ();

    fn next_iter(&mut self) -> Result<ArgminIterationData<Self::Parameters>, Error> {
        let param = self.cur_param();
        let grad = self.gradient(&param)?;
        let new_param = param.scaled_sub(self.omega, grad);
        let out = ArgminIterationData::new(new_param, 0.0);
        Ok(out)
    }
}
