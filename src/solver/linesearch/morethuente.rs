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

#[derive(Default)]
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
    init_cost_b: Option<f64>,
    /// initial gradient (builder)
    init_grad_b: Option<T>,
    /// Search direction (builder)
    search_direction_b: Option<T>,
    /// initial parameter vector
    init_param: T,
    /// initial cost
    init_cost: f64,
    /// initial gradient
    init_grad: T,
    /// Search direction
    search_direction: T,
    /// Search direction in 1D
    sd: f64,
    /// c1
    c1: f64,
    /// c2
    c2: f64,
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
    /// alpha_l
    alpha_l: f64,
    /// alpha_u
    alpha_u: f64,
    /// alpha_min
    alpha_min: f64,
    /// alpha_max
    alpha_max: f64,
    /// best step
    best_step: Step,
    /// endpoint
    endpoint: Step,
    /// current step
    cur_step: Step,
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
            init_cost_b: None,
            init_grad_b: None,
            search_direction_b: None,
            init_param: T::default(),
            init_cost: std::f64::INFINITY,
            init_grad: T::default(),
            search_direction: T::default(),
            sd: 0.0,
            c1: 1e-4,
            c2: 0.9,
            xtrapf: 4.0,
            width: std::f64::NAN,
            width1: std::f64::NAN,
            xtol: 1e-5,
            alpha: 1.0,
            alpha_l: 0.0,
            alpha_u: std::f64::INFINITY,
            alpha_min: 0.0,
            alpha_max: std::f64::INFINITY,
            best_step: Step::default(),
            endpoint: Step::default(),
            cur_step: Step::default(),
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
        self.c1 = c1;
        self.c2 = c2;
        Ok(self)
    }
    // /// Set delta
    // pub fn set_delta(&mut self, delta: f64) -> Result<&mut Self, Error> {
    //     if delta <= 0.0 || delta >= 1.0 {
    //         return Err(ArgminError::InvalidParameter {
    //             parameter: "MoreThuenteLineSearch: Parameter delta must >1.".to_string(),
    //         }.into());
    //     }
    //     self.delta = delta;
    //     Ok(self)
    // }

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
        self.alpha_min = alpha_min;
        self.alpha_max = alpha_max;
        Ok(self)
    }

    /// Set initial cost function value
    pub fn set_initial_cost(&mut self, init_cost: f64) -> &mut Self {
        self.init_cost_b = Some(init_cost);
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
        self.init_cost_b = Some(self.apply(&tmp)?);
        Ok(self)
    }

    /// Calculate initial cost function value
    pub fn calc_inital_gradient(&mut self) -> Result<&mut Self, Error> {
        let tmp = self.base.cur_param();
        self.init_grad_b = Some(self.gradient(&tmp)?);
        Ok(self)
    }

    // fn psi(&mut self, alpha: f64) -> Result<f64, Error> {
    //     let new_param = self
    //         .init_param
    //         .scaled_add(alpha, self.search_direction.clone());
    //     Ok(self.apply(&new_param)?
    //         - self.init_cost
    //         - self.mu * alpha * self.init_grad.dot(self.search_direction.clone()))
    // }

    // fn psi_deriv(&mut self, alpha: f64) -> Result<f64, Error> {
    //     let new_param = self
    //         .init_param
    //         .scaled_add(alpha, self.search_direction.clone());
    //     Ok(self
    //         .gradient(&new_param)?
    //         .dot(self.search_direction.clone())
    //         - self.mu * self.init_grad.dot(self.search_direction.clone()))
    // }
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

        self.init_cost = match self.init_cost_b {
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

        self.sd = self.init_grad.dot(self.search_direction.clone());

        // compute search direction in 1D
        if self.sd >= 0.0 {
            return Err(ArgminError::ConditionViolated {
                text: "MoreThuenteLineSearch: Search direction must be a descent direction."
                    .to_string(),
            }.into());
        }

        self.width = self.alpha_max - self.alpha_min;
        self.width1 = 2.0 * self.width;

        self.best_step = Step::new(0.0, self.init_cost, self.sd);
        self.endpoint = Step::new(0.0, self.init_cost, self.sd);

        Ok(())
    }

    fn next_iter(&mut self) -> Result<ArgminIterationData<Self::Parameters>, Error> {
        let dgtest = self.c1 * self.sd;

        // set the minimum and maximum steps to correspond to the present interval of uncertainty
        if self.brackt {
            self.alpha_min = self.alpha_l.min(self.alpha_u);
            self.alpha_max = self.alpha_l.max(self.alpha_u);
        } else {
            self.alpha_min = self.alpha;
            self.alpha_max = self.cur_step.x + self.xtrapf * (self.alpha - self.alpha_l);
        }

        // alpha needs to be within bounds
        self.alpha = self.alpha.max(self.alpha_min);
        self.alpha = self.alpha.min(self.alpha_max);

        // If an unusual termination is to occur then let alpha be the lowest point obtained so
        // far.
        if (self.brackt && (self.alpha <= self.alpha_min || self.alpha >= self.alpha_max))
            || (self.brackt && (self.alpha_max - self.alpha_min) <= self.xtol * self.alpha_max)
        {
            self.alpha = self.best_step.x;
        }

        // Evaluate the function and gradient at new alpha and compute the directional derivative
        let new_param = self
            .init_param
            .scaled_add(self.alpha, self.search_direction.clone());
        let new_cost = self.apply(&new_param)?;
        let new_grad = self.gradient(&new_param)?;
        self.base.set_cur_cost(new_cost);
        self.base.set_cur_param(new_param);
        let dg = self.search_direction.dot(new_grad);
        let ftest1 = self.base.cur_cost() + self.alpha * dgtest;

        // Calling terminate here myself
        // self.terminate();
        // if self.base.terminated() {
        //     let out = ArgminIterationData::new(new_param, cur_cost);
        //     return Ok(out);
        // }

        if self.stage1 && self.base.cur_cost() <= ftest1 && dg >= self.c1.min(self.c2) * self.sd {
            self.stage1 = false;
        }

        if self.stage1 && self.base.cur_cost() <= self.init_cost && self.base.cur_cost() > ftest1 {
            unimplemented!();
        } else {
            unimplemented!();
        }

        if self.brackt {
            unimplemented!();
        }

        unimplemented!()
        // let new_param = self
        //     .init_param
        //     .scaled_add(self.alpha, self.search_direction.clone());
        // let cur_cost = self.apply(&new_param)?;
        //
        // let alpha = self.alpha;
        // let alpha_l = self.alpha_l;
        // let alpha_u = self.alpha_u;
        // let psi_a = self.psi(alpha)?;
        // let psi_al = self.psi(alpha_l)?;
        // let d_psi_a = self.psi_deriv(alpha)? * (alpha_l - alpha);
        //
        // // Case U1:
        // if psi_a > psi_al {
        //     self.alpha_l = alpha_l;
        //     self.alpha_u = alpha;
        // }
        // // Case U2
        // if psi_a <= psi_al && d_psi_a > 0.0 {
        //     self.alpha_l = alpha;
        //     self.alpha_u = alpha_u;
        // }
        // // Case U3
        // if psi_a <= psi_al && d_psi_a < 0.0 {
        //     self.alpha_l = alpha;
        //     self.alpha_u = alpha_l;
        // }
        //
        // let alpha_candidate = self.alpha + self.delta * (self.alpha - self.alpha_l);
        // self.alpha = if alpha_candidate < self.alpha_max {
        //     alpha_candidate
        // } else {
        //     self.alpha_max
        // };
        //
        // let out = ArgminIterationData::new(new_param, cur_cost);
        // Ok(out)
    }
}
