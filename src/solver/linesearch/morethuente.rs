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
pub struct MoreThuenteLineSearch<T>
where
    T: std::default::Default
        + Clone
        + ArgminSub<T>
        + ArgminDot<T, f64>
        + ArgminScaledAdd<T, f64>
        + ArgminScaledSub<T, f64>,
{
    /// initial parameter vector
    init_param: Option<T>,
    /// initial cost
    init_cost: Option<f64>,
    /// initial gradient
    init_grad: Option<T>,
    /// Search direction
    search_direction: Option<T>,
    /// Search direction in 1D
    sd: f64,
    /// mu
    mu: f64,
    /// delta
    delta: f64,
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
            init_param: None,
            init_cost: None,
            init_grad: None,
            search_direction: None,
            sd: 0.0,
            mu: 0.9,
            delta: 1.1,
            alpha: 1.0,
            alpha_l: 0.0,
            alpha_u: std::f64::INFINITY,
            alpha_min: 0.0,
            alpha_max: std::f64::INFINITY,
            base: ArgminBase::new(operator, T::default()),
        }
    }

    /// Set search direction
    pub fn set_search_direction(&mut self, search_direction: T) -> &mut Self {
        self.search_direction = Some(search_direction);
        self
    }

    /// Set initial parameter
    pub fn set_initial_parameter(&mut self, param: T) -> &mut Self {
        self.init_param = Some(param.clone());
        self.base.set_cur_param(param);
        self
    }

    /// set current gradient value
    pub fn set_cur_grad(&mut self, grad: T) -> &mut Self {
        self.base.set_cur_grad(grad);
        self
    }

    /// Set mu
    pub fn set_mu(&mut self, mu: f64) -> Result<&mut Self, Error> {
        if mu <= 0.0 || mu >= 1.0 {
            return Err(ArgminError::InvalidParameter {
                parameter: "MoreThuenteLineSearch: Parameter mu must be in (0, 1).".to_string(),
            }.into());
        }
        self.mu = mu;
        Ok(self)
    }

    /// Set delta
    pub fn set_delta(&mut self, delta: f64) -> Result<&mut Self, Error> {
        if delta <= 0.0 || delta >= 1.0 {
            return Err(ArgminError::InvalidParameter {
                parameter: "MoreThuenteLineSearch: Parameter delta must >1.".to_string(),
            }.into());
        }
        self.delta = delta;
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
        self.alpha_min = alpha_min;
        self.alpha_max = alpha_max;
        Ok(self)
    }

    /// Set initial cost function value
    pub fn set_initial_cost(&mut self, init_cost: f64) -> &mut Self {
        self.init_cost = Some(init_cost);
        self
    }

    /// Set initial gradient
    pub fn set_initial_gradient(&mut self, init_grad: T) -> &mut Self {
        self.init_grad = Some(init_grad);
        self
    }

    /// Calculate initial cost function value
    pub fn calc_inital_cost(&mut self) -> Result<&mut Self, Error> {
        let tmp = self.base.cur_param();
        self.init_cost = Some(self.apply(&tmp)?);
        Ok(self)
    }

    /// Calculate initial cost function value
    pub fn calc_inital_gradient(&mut self) -> Result<&mut Self, Error> {
        let tmp = self.base.cur_param();
        self.init_grad = Some(self.gradient(&tmp)?);
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
        match self.init_param {
            None => {
                return Err(ArgminError::NotInitialized {
                    text: "MoreThuenteLineSearch: Initial parameter not initialized. Call `set_initial_parameter`.".to_string(),
                }.into());
            }
            _ => (),
        }

        match self.init_cost {
            None => {
                return Err(ArgminError::NotInitialized {
                    text: "MoreThuenteLineSearch: Initial cost not computed. Call `set_initial_cost` or `calc_inital_cost`.".to_string(),
                }.into());
            }
            _ => (),
        }

        match self.init_grad {
            None => {
                return Err(ArgminError::NotInitialized {
                    text: "MoreThuenteLineSearch: Initial gradient not computed. Call `set_initial_grad` or `calc_inital_grad`.".to_string(),
                }.into());
            }
            _ => (),
        }

        match self.search_direction {
            None => {
                return Err(ArgminError::NotInitialized {
                    text: "MoreThuenteLineSearch: Search direction not initialized. Call `set_search_direction`.".to_string(),
                }.into());
            }
            _ => (),
        }

        self.sd = (self.init_grad.as_ref().unwrap().clone())
            .dot(self.search_direction.as_ref().unwrap().clone());

        Ok(())
    }

    fn next_iter(&mut self) -> Result<ArgminIterationData<Self::Parameters>, Error> {
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

        unimplemented!()
    }
}
