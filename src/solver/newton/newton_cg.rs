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
        + ArgminNorm<f64>
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
        + ArgminNorm<f64>
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
        + ArgminNorm<f64>
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

        // Solve CG subproblem
        let op: CGSubProblem<'a, T, H> = CGSubProblem::new(hessian.clone());
        let cg_op = Box::new(op);

        let mut x_p = param.zero();
        let mut x: T = param.zero();
        let mut cg = ConjugateGradient::new(cg_op, grad.scale(-1.0), x_p.clone())?;

        cg.init()?;
        let norm = grad.norm();
        for iter in 0.. {
            let data = self.next_iter()?;
            cg.increment_iter();
            cg.set_cur_param(data.param());
            cg.set_cur_cost(data.cost());
            let p = cg.p();
            let curvature = p.dot(hessian.dot(p.clone()));
            if curvature <= 0.0 {
                if iter == 0 {
                    x = data.param();
                    break;
                } else {
                    x = x_p;
                    break;
                }
            }
            if cg.residual().norm() <= 0.5f64.min(norm.sqrt()) * norm {
                x = data.param();
                break;
            }
            x_p = x.clone();
        }

        // let new_param = param.scaled_sub(self.gamma, hessian.ainv()?.dot(grad));
        unimplemented!()
        // let out = ArgminIterationData::new(new_param, 0.0);
        // Ok(out)
    }
}

#[derive(Clone)]
struct CGSubProblem<'a, T, H>
where
    H: 'a + Clone + Default + ArgminDot<T, T>,
    T: 'a + Clone,
{
    hessian: H,
    phantom: std::marker::PhantomData<&'a T>,
}

impl<'a, T, H> CGSubProblem<'a, T, H>
where
    H: 'a + Clone + Default + ArgminDot<T, T>,
    T: 'a + Clone,
{
    /// constructor
    pub fn new(hessian: H) -> Self {
        CGSubProblem {
            hessian: hessian,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<'a, T, H> ArgminOperator for CGSubProblem<'a, T, H>
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
