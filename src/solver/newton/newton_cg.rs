// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Newton-CG method

use crate::solver::conjugategradient::ConjugateGradient;
use prelude::*;
use std;
use std::default::Default;

/// Newton-CG Method
#[derive(ArgminSolver)]
pub struct NewtonCG<'a, T, H>
where
    T: 'a
        + Clone
        + Default
        + ArgminScaledSub<T, f64>
        + ArgminScaledAdd<T, f64>
        + ArgminDot<T, f64>
        + ArgminAdd<T>
        + ArgminSub<T>
        + ArgminZero
        + ArgminScale<f64>,
    H: 'a + Clone + Default + ArgminInv<H> + ArgminDot<T, T>,
{
    // /// CG
    // cg: ConjugateGradient<'a, T>,
    /// Base stuff
    base: ArgminBase<'a, T, f64, H>,
}

impl<'a, T, H> NewtonCG<'a, T, H>
where
    T: 'a
        + Clone
        + Default
        + ArgminScaledSub<T, f64>
        + ArgminScaledAdd<T, f64>
        + ArgminDot<T, f64>
        + ArgminAdd<T>
        + ArgminSub<T>
        + ArgminZero
        + ArgminScale<f64>,
    H: 'a + Clone + Default + ArgminInv<H> + ArgminDot<T, T>,
{
    /// Constructor
    pub fn new(
        cost_function: Box<ArgminOperator<Parameters = T, OperatorOutput = f64, Hessian = H> + 'a>,
        init_param: T,
    ) -> Self {
        NewtonCG {
            base: ArgminBase::new(cost_function, init_param),
        }
    }
}

impl<'a, T, H> ArgminNextIter for NewtonCG<'a, T, H>
where
    T: 'a
        + Clone
        + Default
        + ArgminScaledSub<T, f64>
        + ArgminScaledAdd<T, f64>
        + ArgminDot<T, f64>
        + ArgminAdd<T>
        + ArgminSub<T>
        + ArgminZero
        + ArgminScale<f64>,
    H: 'a + Clone + Default + ArgminInv<H> + ArgminDot<T, T>,
{
    type Parameters = T;
    type OperatorOutput = f64;
    type Hessian = H;

    fn next_iter(&mut self) -> Result<ArgminIterationData<Self::Parameters>, Error> {
        let param = self.cur_param();
        let grad = self.gradient(&param)?;
        let hessian = self.hessian(&param)?;

        let op: CGProblem<'a, T, H> = CGProblem::new(hessian);
        let bla = Box::new(op);

        let cg = ConjugateGradient::new(bla, grad, param.zero());

        // let cg = ConjugateGradient::new(op, b, self.p.zero());
        // let new_param = param.scaled_sub(self.gamma, hessian.ainv()?.dot(grad));
        unimplemented!()
        // let out = ArgminIterationData::new(new_param, 0.0);
        // Ok(out)
    }
}

#[derive(Clone)]
struct CGProblem<'a, T, H>
where
    H: 'a + Clone + Default + ArgminDot<T, T>,
    T: 'a + Clone,
{
    hessian: H,
    phantom: std::marker::PhantomData<&'a T>,
}

impl<'a, T, H> CGProblem<'a, T, H>
where
    H: 'a + Clone + Default + ArgminDot<T, T>,
    T: 'a + Clone,
{
    /// constructor
    pub fn new(hessian: H) -> Self {
        CGProblem {
            hessian: hessian,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<'a, T, H> ArgminOperator for CGProblem<'a, T, H>
where
    T: 'a + Clone,
    H: 'a + Clone + Default + ArgminDot<T, T>,
{
    type Parameters = T;
    type OperatorOutput = T;
    type Hessian = ();

    fn apply(&self, p: &T) -> Result<T, Error> {
        Ok(self.hessian.dot(p.clone()))
    }

    /// dont ever clone this
    fn box_clone(&self) -> Box<ArgminOperator<Parameters = T, OperatorOutput = T, Hessian = ()>> {
        unimplemented!()
    }
}
