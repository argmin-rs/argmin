// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Trust region solver
//!
//! ## Reference
//!
//! TODO
//!
// //!
// //! # Example
// //!
// //! ```rust
// //! todo
// //! ```

use prelude::*;
use solver::trustregion::reduction_ratio;
// use solver::trustregion::CauchyPoint;
// use solver::trustregion::Dogleg;
use solver::trustregion::Steihaug;
use std;

/// Trust region solver
#[derive(ArgminSolver)]
pub struct TrustRegion<'a, T, H>
where
    T: Clone
        + std::default::Default
        + std::fmt::Debug
        + ArgminWeightedDot<T, f64, H>
        + ArgminDot<T, f64>
        + ArgminNorm<f64>
        + ArgminAdd<T>
        + ArgminSub<T>
        + ArgminZero
        + ArgminScale<f64>,
    H: Clone + std::default::Default + ArgminInv<H> + ArgminDot<T, T>,
{
    /// Radius
    radius: f64,
    /// Maximum Radius
    max_radius: f64,
    /// eta \in [0, 1/4)
    eta: f64,
    /// subproblem
    subproblem: Box<ArgminTrustRegion<Parameters = T, OperatorOutput = f64, Hessian = H> + 'a>,
    /// f(xk)
    fxk: f64,
    /// mk(0)
    mk0: f64,
    /// base
    base: ArgminBase<'a, T, f64, H>,
}

impl<'a, T, H> TrustRegion<'a, T, H>
where
    T: 'a
        + Clone
        + std::default::Default
        + std::fmt::Debug
        + ArgminWeightedDot<T, f64, H>
        + ArgminDot<T, f64>
        + ArgminNorm<f64>
        + ArgminAdd<T>
        + ArgminSub<T>
        + ArgminZero
        + ArgminScale<f64>,
    H: 'a + Clone + std::default::Default + ArgminInv<H> + ArgminDot<T, T>,
{
    /// Constructor
    ///
    /// Parameters:
    ///
    /// `operator`: operator
    pub fn new(
        operator: Box<ArgminOperator<Parameters = T, OperatorOutput = f64, Hessian = H>>,
        param: T,
    ) -> Self {
        let base = ArgminBase::new(operator.clone(), param);
        // let subproblem = Box::new(CauchyPoint::new(operator.clone()));
        // let subproblem = Box::new(Dogleg::new(operator.clone()));
        let mut subproblem = Box::new(Steihaug::new(operator.clone()));
        subproblem.set_max_iters(2);
        TrustRegion {
            radius: 1.0,
            max_radius: 100.0,
            eta: 0.125,
            subproblem: subproblem,
            fxk: std::f64::NAN,
            mk0: std::f64::NAN,
            base: base,
        }
    }

    /// set radius
    pub fn set_radius(&mut self, radius: f64) -> &mut Self {
        self.radius = radius;
        self
    }

    /// Set maximum radius
    pub fn set_max_radius(&mut self, max_radius: f64) -> &mut Self {
        self.max_radius = max_radius;
        self
    }

    /// Set eta
    pub fn set_eta(&mut self, eta: f64) -> Result<&mut Self, Error> {
        if eta >= 0.25 || eta < 0.0 {
            return Err(ArgminError::InvalidParameter {
                text: "TrustRegion: eta must be in [0, 1/4).".to_string(),
            }
            .into());
        }
        self.eta = eta;
        Ok(self)
    }

    fn m(&self, p: &T) -> f64 {
        self.fxk
            + p.dot(self.base.cur_grad())
            + 0.5 * p.weighted_dot(self.base.cur_hessian(), p.clone())
    }
}

impl<'a, T, H> ArgminNextIter for TrustRegion<'a, T, H>
where
    T: Clone
        + std::default::Default
        + std::fmt::Debug
        + ArgminWeightedDot<T, f64, H>
        + ArgminDot<T, f64>
        + ArgminNorm<f64>
        + ArgminAdd<T>
        + ArgminSub<T>
        + ArgminZero
        + ArgminScale<f64>,
    H: Clone + std::default::Default + ArgminInv<H> + ArgminDot<T, T>,
{
    type Parameters = T;
    type OperatorOutput = f64;
    type Hessian = H;

    fn init(&mut self) -> Result<(), Error> {
        let param = self.base.cur_param();
        let grad = self.gradient(&param)?;
        self.base.set_cur_grad(grad);
        let hessian = self.hessian(&param)?;
        self.base.set_cur_hessian(hessian);
        self.fxk = self.apply(&param)?;
        self.mk0 = self.fxk;
        Ok(())
    }

    fn next_iter(&mut self) -> Result<ArgminIterationData<Self::Parameters>, Error> {
        self.subproblem.set_grad(self.base.cur_grad());
        self.subproblem.set_hessian(self.base.cur_hessian());
        self.subproblem.set_radius(self.radius);
        let pk = self.subproblem.run_fast()?.param;
        let new_param = pk.add(self.base.cur_param().clone());
        let fxkpk = self.apply(&new_param)?;
        let mkpk = self.m(&pk);
        let rho = reduction_ratio(self.fxk, fxkpk, self.mk0, mkpk);

        let pk_norm = pk.norm();

        let cur_radius = self.radius;
        self.radius = if rho < 0.25 {
            0.25 * pk_norm
        } else {
            // if rho > 0.75 && pk_norm == self.radius {
            if rho > 0.75 && (pk_norm - self.radius).abs() <= 10.0 * std::f64::EPSILON {
                self.max_radius.min(2.0 * self.radius)
            } else {
                self.radius
            }
        };

        let mut out = if rho > self.eta {
            // self.base.set_cur_param(new_param.clone());
            self.fxk = fxkpk;
            self.mk0 = fxkpk;
            let grad = self.gradient(&new_param)?;
            self.base.set_cur_grad(grad);
            let hessian = self.hessian(&new_param)?;
            self.base.set_cur_hessian(hessian);
            ArgminIterationData::new(new_param, fxkpk)
        } else {
            ArgminIterationData::new(self.base.cur_param(), self.fxk)
        };
        let kv = make_kv!("radius" => cur_radius;);
        out.add_kv(kv);

        return Ok(out);
    }
}
