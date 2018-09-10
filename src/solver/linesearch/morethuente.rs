// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # More-Thuente line search algorithm
//!
//! TODO: Proper documentation.
//!
//! This implementation follows the excellent MATLAB implementation of Dianne P. O'Leary at
//! http://www.cs.umd.edu/users/oleary/software/
//!
// //!
// //! # Example
// //!
// //! ```rust
// //! todo
// //! ```

use prelude::*;
use std;

#[derive(Default, Clone)]
struct Step {
    pub x: f64,
    pub fx: f64,
    pub gx: f64,
}

impl Step {
    pub fn new(x: f64, fx: f64, gx: f64) -> Self {
        Step { x, fx, gx }
    }
}

/// More-Thuente Line Search
#[derive(ArgminSolver)]
pub struct MoreThuenteLineSearch<T>
where
    T: std::default::Default
        + Clone
        + ArgminSub<T>
        + ArgminDot<T, f64>
        + ArgminScaledAdd<T, f64>
        + ArgminScaledSub<T, f64>,
{
    /// initial parameter vector (builder)
    init_param_b: Option<T>,
    /// initial cost (builder)
    finit_b: Option<f64>,
    /// initial gradient (builder)
    init_grad_b: Option<T>,
    /// Search direction (builder)
    search_direction_b: Option<T>,
    /// initial parameter vector
    init_param: T,
    /// initial cost
    finit: f64,
    /// initial gradient
    init_grad: T,
    /// Search direction
    search_direction: T,
    /// Search direction in 1D
    dginit: f64,
    /// dgtest
    dgtest: f64,
    /// c1
    ftol: f64,
    /// c2
    gtol: f64,
    /// xtrapf?
    xtrapf: f64,
    /// width of interval
    width: f64,
    /// width of what?
    width1: f64,
    /// xtol
    xtol: f64,
    /// alpha
    alpha: f64,
    /// stpmin
    stmin: f64,
    /// stpmax
    stmax: f64,
    /// current step
    stp: Step,
    /// stx
    stx: Step,
    /// sty
    sty: Step,
    /// bracketed
    brackt: bool,
    /// stage1
    stage1: bool,
    /// infoc
    infoc: bool,
    /// base
    base: ArgminBase<T, f64>,
}

impl<T> MoreThuenteLineSearch<T>
where
    T: std::default::Default
        + Clone
        + ArgminSub<T>
        + ArgminDot<T, f64>
        + ArgminScaledAdd<T, f64>
        + ArgminScaledSub<T, f64>,
    MoreThuenteLineSearch<T>: ArgminSolver<Parameters = T, OperatorOutput = f64>,
{
    /// Constructor
    ///
    /// Parameters:
    ///
    /// `cost_function`: cost function
    /// `mu`: todo
    pub fn new(operator: Box<ArgminOperator<Parameters = T, OperatorOutput = f64>>) -> Self {
        MoreThuenteLineSearch {
            init_param_b: None,
            finit_b: None,
            init_grad_b: None,
            search_direction_b: None,
            init_param: T::default(),
            finit: std::f64::INFINITY,
            init_grad: T::default(),
            search_direction: T::default(),
            dginit: 0.0,
            dgtest: 0.0,
            ftol: 1e-4,
            gtol: 0.9,
            xtrapf: 4.0,
            width: std::f64::NAN,
            width1: std::f64::NAN,
            xtol: 1e-5,
            alpha: 1.0,
            stmin: 0.0,
            stmax: std::f64::INFINITY,
            stp: Step::default(),
            stx: Step::default(),
            sty: Step::default(),
            brackt: false,
            stage1: true,
            infoc: true,
            base: ArgminBase::new(operator, T::default()),
        }
    }

    /// Set search direction
    pub fn set_search_direction(&mut self, search_direction: T) -> &mut Self {
        self.search_direction_b = Some(search_direction);
        self
    }

    /// Set initial parameter
    pub fn set_initial_parameter(&mut self, param: T) -> &mut Self {
        self.init_param_b = Some(param.clone());
        self.base.set_cur_param(param);
        self
    }

    /// set current gradient value
    pub fn set_cur_grad(&mut self, grad: T) -> &mut Self {
        self.base.set_cur_grad(grad);
        self
    }

    /// Set mu
    pub fn set_c(&mut self, c1: f64, c2: f64) -> Result<&mut Self, Error> {
        if c1 <= 0.0 || c1 >= c2 {
            return Err(ArgminError::InvalidParameter {
                parameter: "MoreThuenteLineSearch: Parameter c1 must be in (0, c2).".to_string(),
            }.into());
        }
        if c2 <= c1 || c2 >= 1.0 {
            return Err(ArgminError::InvalidParameter {
                parameter: "MoreThuenteLineSearch: Parameter c2 must be in (c1, 1).".to_string(),
            }.into());
        }
        self.ftol = c1;
        self.gtol = c2;
        Ok(self)
    }

    /// Set initial alpha value
    pub fn set_initial_alpha(&mut self, alpha: f64) -> Result<&mut Self, Error> {
        if alpha <= 0.0 {
            return Err(ArgminError::InvalidParameter {
                parameter: "MoreThuenteLineSearch: Inital alpha must be > 0.".to_string(),
            }.into());
        }
        self.alpha = alpha;
        Ok(self)
    }

    /// set alpha limits
    pub fn set_alpha_min_max(
        &mut self,
        alpha_min: f64,
        alpha_max: f64,
    ) -> Result<&mut Self, Error> {
        if alpha_min < 0.0 {
            return Err(ArgminError::InvalidParameter {
                parameter: "MoreThuenteLineSearch: alpha_min must be >= 0.0.".to_string(),
            }.into());
        }
        if alpha_max <= alpha_min {
            return Err(ArgminError::InvalidParameter {
                parameter: "MoreThuenteLineSearch: alpha_min must be smaller than alpha_max."
                    .to_string(),
            }.into());
        }
        self.stmin = alpha_min;
        self.stmax = alpha_max;
        Ok(self)
    }

    /// Set initial cost function value
    pub fn set_initial_cost(&mut self, init_cost: f64) -> &mut Self {
        self.finit_b = Some(init_cost);
        self
    }

    /// Set initial gradient
    pub fn set_initial_gradient(&mut self, init_grad: T) -> &mut Self {
        self.init_grad_b = Some(init_grad);
        self
    }

    /// Calculate initial cost function value
    pub fn calc_inital_cost(&mut self) -> Result<&mut Self, Error> {
        let tmp = self.base.cur_param();
        self.finit_b = Some(self.apply(&tmp)?);
        Ok(self)
    }

    /// Calculate initial cost function value
    pub fn calc_inital_gradient(&mut self) -> Result<&mut Self, Error> {
        let tmp = self.base.cur_param();
        self.init_grad_b = Some(self.gradient(&tmp)?);
        Ok(self)
    }
}

impl<T> ArgminNextIter for MoreThuenteLineSearch<T>
where
    T: std::default::Default
        + Clone
        + ArgminSub<T>
        + ArgminDot<T, f64>
        + ArgminScaledAdd<T, f64>
        + ArgminScaledSub<T, f64>,
{
    type Parameters = T;
    type OperatorOutput = f64;

    fn init(&mut self) -> Result<(), Error> {
        self.init_param = match self.init_param_b {
            None => {
                return Err(ArgminError::NotInitialized {
                    text: "MoreThuenteLineSearch: Initial parameter not initialized. Call `set_initial_parameter`.".to_string(),
                }.into());
            }
            Some(ref x) => x.clone(),
        };

        self.finit = match self.finit_b {
            None => {
                return Err(ArgminError::NotInitialized {
                    text: "MoreThuenteLineSearch: Initial cost not computed. Call `set_initial_cost` or `calc_inital_cost`.".to_string(),
                }.into());
            }
            Some(ref x) => x.clone(),
        };

        self.init_grad = match self.init_grad_b {
            None => {
                return Err(ArgminError::NotInitialized {
                    text: "MoreThuenteLineSearch: Initial gradient not computed. Call `set_initial_grad` or `calc_inital_grad`.".to_string(),
                }.into());
            }
            Some(ref x) => x.clone(),
        };

        self.search_direction = match self.search_direction_b {
            None => {
                return Err(ArgminError::NotInitialized {
                    text: "MoreThuenteLineSearch: Search direction not initialized. Call `set_search_direction`.".to_string(),
                }.into());
            }
            Some(ref x) => x.clone(),
        };

        self.dginit = self.init_grad.dot(self.search_direction.clone());

        // compute search direction in 1D
        if self.dginit >= 0.0 {
            return Err(ArgminError::ConditionViolated {
                text: "MoreThuenteLineSearch: Search direction must be a descent direction."
                    .to_string(),
            }.into());
        }

        self.dgtest = self.ftol * self.dginit;
        self.width = self.stmax - self.stmin;
        self.width1 = 2.0 * self.width;

        self.stp = Step::new(self.alpha, std::f64::NAN, std::f64::NAN);
        self.stx = Step::new(0.0, self.finit, self.dginit);
        self.sty = Step::new(0.0, self.finit, self.dginit);

        Ok(())
    }

    fn next_iter(&mut self) -> Result<ArgminIterationData<Self::Parameters>, Error> {
        // set the minimum and maximum steps to correspond to the present interval of uncertainty
        if self.brackt {
            self.stmin = self.stx.x.min(self.sty.x);
            self.stmax = self.stx.x.max(self.sty.x);
        } else {
            self.stmin = self.stx.x;
            self.stmax = self.stp.x + self.xtrapf * (self.stp.x - self.stx.x);
        }

        // alpha needs to be within bounds
        self.stp.x = self.stp.x.max(self.stmin);
        self.stp.x = self.stp.x.min(self.stmax);

        // If an unusual termination is to occur then let alpha be the lowest point obtained so
        // far.
        if (self.brackt && (self.stp.x <= self.stmin || self.stp.x >= self.stmax))
            || (self.brackt && (self.stmax - self.stmin) <= self.xtol * self.stmax)
        {
            self.stp.x = self.stx.x;
        }

        // Evaluate the function and gradient at new stp.x and compute the directional derivative
        //
        let new_param = self
            .init_param
            .scaled_add(self.stp.x, self.search_direction.clone());
        let new_cost = self.apply(&new_param)?;
        let new_grad = self.gradient(&new_param)?;
        self.base.set_cur_cost(new_cost);
        self.base.set_cur_param(new_param);
        self.stx.fx = new_cost;
        let dg = self.search_direction.dot(new_grad);
        let ftest1 = self.finit + self.stp.x * self.dgtest;

        // Calling terminate here myself
        // self.terminate();
        // if self.base.terminated() {
        //     let out = ArgminIterationData::new(new_param, cur_cost);
        //     return Ok(out);
        // }

        if self.stage1
            && self.base.cur_cost() <= ftest1
            && dg >= self.ftol.min(self.gtol) * self.dginit
        {
            self.stage1 = false;
        }

        if self.stage1 && self.base.cur_cost() <= self.finit && self.base.cur_cost() > ftest1 {
            let fm = self.base.cur_cost() - self.stp.x * self.dgtest;
            let fxm = self.stx.fx - self.stx.x * self.dgtest;
            let fym = self.sty.fx - self.sty.x * self.dgtest;
            let dgm = dg - self.dgtest;
            let dgxm = self.stx.gx - self.dgtest;
            let dgym = self.sty.gx - self.dgtest;

            let (stx1, sty1, stp1, brackt1, _stmin, _stmax) = cstep(
                Step::new(self.stx.x, fxm, dgxm),
                Step::new(self.sty.x, fym, dgym),
                Step::new(self.stp.x, fm, dgm),
                self.brackt,
                self.stmin,
                self.stmax,
            );

            self.stx.fx = stx1.fx + stx1.x * self.dgtest;
            self.sty.fx = sty1.fx + sty1.x * self.dgtest;
            self.stx.gx = stx1.gx + self.dgtest;
            self.sty.gx = sty1.gx + self.dgtest;
            self.brackt = brackt1;
            self.stp = stp1;
        } else {
            let (stx1, sty1, stp1, brackt1, _stmin, _stmax) = cstep(
                self.stx.clone(),
                self.sty.clone(),
                self.stp.clone(),
                self.brackt,
                self.stmin,
                self.stmax,
            );
            self.stx = stx1;
            self.sty = sty1;
            self.stp = stp1;
            self.brackt = brackt1;
        }

        if self.brackt {
            if (self.sty.x - self.stx.x).abs() >= 0.66 * self.width1 {
                self.stp.x = self.stx.x + 0.5 * (self.sty.x - self.stx.x);
            }
            self.width1 = self.width;
            self.width = (self.sty.x - self.stx.x).abs();
        }

        let out = ArgminIterationData::new(self.base.cur_param(), self.base.cur_cost());
        Ok(out)
    }
}

fn cstep(
    stx: Step,
    sty: Step,
    stp: Step,
    brackt: bool,
    stpmin: f64,
    stpmax: f64,
) -> (Step, Step, Step, bool, f64, f64) {
    // let mut info: i8 = 0;
    let bound: bool;
    let mut stpf: f64;
    let stpc: f64;
    let stpq: f64;
    let mut brackt = brackt;

    // check inputs
    if (brackt && (stp.x <= stx.x.min(sty.x) || stp.x >= stx.x.max(sty.x)))
        || stx.gx * (stp.x - stx.x) >= 0.0
        || stpmax < stpmin
    {
        return (stx, sty, stp, brackt, stpmin, stpmax);
    }

    // determine if the derivatives have opposite sign
    let sgnd = stp.gx * (stx.gx / stx.gx.abs());

    if stp.fx > stx.fx {
        // First case. A higher function value. The minimum is bracketed. If the cubic step is closer to
        // stx.x than the quadratic step, the cubic step is taken, else the average of the cubic and
        // the quadratic steps is taken.
        // info = 1;
        bound = true;
        let theta = 3.0 * (stx.fx - stp.fx) / (stp.x - stx.x) + stx.gx + stp.gx;
        let tmp = vec![theta, stx.gx, stp.gx];
        let s = tmp.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let mut gamma = s * ((theta / s).powi(2) - (stx.gx / s) * (stp.gx / s)).sqrt();
        if stp.x < stx.x {
            gamma = -gamma;
        }

        let p = (gamma - stx.gx) + theta;
        let q = ((gamma - stx.gx) + gamma) + stp.gx;
        let r = p / q;
        stpc = stx.x + r * (stp.x - stx.x);
        stpq = stx.x
            + ((stx.gx / ((stx.fx - stp.fx) / (stp.x - stx.x) + stx.gx)) / 2.0) * (stp.x - stx.x);
        if (stpc - stx.x).abs() < (stpq - stx.x).abs() {
            stpf = stpc;
        } else {
            stpf = stpc + (stpq - stpc) / 2.0;
        }
        brackt = true;
    } else if sgnd < 0.0 {
        // Second case. A lower function value and derivatives of opposite sign. The minimum is
        // bracketed. If the cubic step is closer to stx.x than the quadtratic (secant) step, the
        // cubic step is taken, else the quadratic step is taken.
        // info = 2;
        bound = false;
        let theta = 3.0 * (stx.fx - stp.fx) / (stp.x - stx.x) + stx.gx + stp.gx;
        let tmp = vec![theta, stx.gx, stp.gx];
        let s = tmp.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let mut gamma = s * ((theta / s).powi(2) - (stx.gx / s) * (stp.gx / s)).sqrt();
        if stp.x > stx.x {
            gamma = -gamma;
        }
        let p = (gamma - stp.gx) + theta;
        let q = ((gamma - stp.gx) + gamma) + stx.gx;
        let r = p / q;
        stpc = stp.x + r * (stx.x - stp.x);
        stpq = stp.x + (stp.gx / (stp.gx - stx.gx)) * (stx.x - stp.x);
        if (stpc - stp.x).abs() > (stpq - stp.x).abs() {
            stpf = stpc;
        } else {
            stpf = stpq;
        }
        brackt = true;
    } else if stp.gx.abs() < stx.gx.abs() {
        // Third case. A lower function value, derivatives of the same sign, and the magnitude of
        // the derivative decreases. The cubic step is only used if the cubic tends to infinity in
        // the direction of the step or if the minimum of the cubic is beyond stp.x. Otherwise the
        // cubic step is defined to be either stpmin or stpmax. The quadtratic (secant) step is
        // also computed and if the minimum is bracketed then the step closest to stx.x is taken,
        // else the step farthest away is taken.
        // info = 3;
        bound = true;
        let theta = 3.0 * (stx.fx - stp.fx) / (stp.x - stx.x) + stx.gx + stp.gx;
        let tmp = vec![theta, stx.gx, stp.gx];
        let s = tmp.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        // the case gamma == 0 only arises if the cubic does not tend to infinity in the direction
        // of the step.

        let mut gamma = s * 0.0f64
            .max((theta / s).powi(2) - (stx.gx / s) * (stp.gx / s))
            .sqrt();
        if stp.x > stx.x {
            gamma = -gamma;
        }

        let p = (gamma - stp.gx) + theta;
        let q = (gamma + (stx.gx - stp.gx)) + gamma;
        let r = p / q;
        if r < 0.0 && gamma != 0.0 {
            stpc = stp.x + r * (stx.x - stp.x);
        } else if stp.x > stx.x {
            stpc = stpmax;
        } else {
            stpc = stpmin;
        }
        stpq = stp.x + (stp.gx / (stp.gx - stx.gx)) * (stx.x - stp.x);
        if brackt {
            if (stp.x - stpc).abs() < (stp.x - stpq).abs() {
                stpf = stpc;
            } else {
                stpf = stpq;
            }
        } else {
            if (stp.x - stpc).abs() > (stp.x - stpq).abs() {
                stpf = stpc;
            } else {
                stpf = stpq;
            }
        }
    } else {
        // Fourth case. A lower function value, derivatives of the same sign, and the magnitued of
        // the derivative does not decrease. If the minimum is not bracketed, the step is either
        // stpmin or stpmax, else the cubic step is taken.
        // info = 4;
        bound = false;
        if brackt {
            let theta = 3.0 * (stp.fx - sty.fx) / (sty.x - stp.x) + sty.gx + stp.gx;
            let tmp = vec![theta, sty.gx, stp.gx];
            let s = tmp.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            let mut gamma = s * ((theta / s).powi(2) - (sty.gx / s) * (stp.gx / s)).sqrt();
            if stp.x > sty.x {
                gamma = -gamma;
            }
            let p = (gamma - stp.gx) + theta;
            let q = ((gamma - stp.gx) + gamma) + sty.gx;
            let r = p / q;
            stpc = stp.x + r * (sty.x - stp.x);
            stpf = stpc;
        } else if stp.x > stx.x {
            stpf = stpmax;
        } else {
            stpf = stpmin;
        }
    }
    // Update the interval of uncertainty. This update does not depend on the new step or the case
    // analysis above.

    let mut stx_o = stx.clone();
    let mut sty_o = sty.clone();
    let mut stp_o = stp.clone();
    if stp.fx > stx.fx {
        sty_o = Step::new(stp.x, stp.fx, stp.gx);
    } else {
        if sgnd < 0.0 {
            sty_o = Step::new(stx.x, stx.fx, stx.gx);
        }
        stx_o = Step::new(stp.x, stp.fx, stp.gx);
    }

    // compute the new step and safeguard it.

    stpf = stpmax.min(stpf);
    stpf = stpmin.max(stpf);

    stp_o.x = stpf;
    if brackt && bound {
        if sty_o.x > stx_o.x {
            stp_o.x = stp.x.min(stx.x + 0.66 * (sty_o.x - stx_o.x));
        } else {
            stp_o.x = stp.x.max(stx.x + 0.66 * (sty_o.x - stx_o.x));
        }
    }

    (stx_o, sty_o, stp_o, brackt, stpmin, stpmax)
}
