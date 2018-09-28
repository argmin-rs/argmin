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
    /// delta: (0, 0.5), used in the Wolve conditions
    delta: f64,
    /// sigma: [delta, 1), used in the Wolfe conditions
    sigma: f64,
    /// epsilon: [0, infinity), used in the approximate Wolfe termination
    epsilon: f64,
    /// theta: (0, 1), used in the update rules when the potential intervals [a, c] or [c, b]
    /// viloate the opposite slope condition
    theta: f64,
    /// gamma: (0, 1), determines when a bisection step is performed
    gamma: f64,
    /// eta: (0, infinity), used in the lower bound for beta_k^N
    eta: f64,
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
            delta: 0.1,
            sigma: 0.9,
            epsilon: 10e-6,
            theta: 0.5,
            gamma: 0.66,
            eta: 0.01,
            base: ArgminBase::new(operator, T::default()),
        }
    }

    /// set current gradient value
    pub fn set_cur_grad(&mut self, grad: T) -> &mut Self {
        self.base.set_cur_grad(grad);
        self
    }

    /// set delta
    pub fn set_delta(&mut self, delta: f64) -> Result<&mut Self, Error> {
        if delta <= 0.0 {
            return Err(ArgminError::InvalidParameter {
                parameter: "HagerZhangLineSearch: delta must be > 0.0.".to_string(),
            }
            .into());
        }
        if delta >= 1.0 {
            return Err(ArgminError::InvalidParameter {
                parameter: "HagerZhangLineSearch: delta must be < 1.0.".to_string(),
            }
            .into());
        }
        self.delta = delta;
        Ok(self)
    }

    /// set sigma
    pub fn set_sigma(&mut self, sigma: f64) -> Result<&mut Self, Error> {
        if sigma < self.delta {
            return Err(ArgminError::InvalidParameter {
                parameter: "HagerZhangLineSearch: sigma must be >= delta.".to_string(),
            }
            .into());
        }
        if sigma >= 1.0 {
            return Err(ArgminError::InvalidParameter {
                parameter: "HagerZhangLineSearch: sigma must be < 1.0.".to_string(),
            }
            .into());
        }
        self.sigma = sigma;
        Ok(self)
    }

    /// set epsilon
    pub fn set_epsilon(&mut self, epsilon: f64) -> Result<&mut Self, Error> {
        if epsilon < 0.0 {
            return Err(ArgminError::InvalidParameter {
                parameter: "HagerZhangLineSearch: epsilon must be >= 0.0.".to_string(),
            }
            .into());
        }
        self.epsilon = epsilon;
        Ok(self)
    }

    /// set theta
    pub fn set_theta(&mut self, theta: f64) -> Result<&mut Self, Error> {
        if theta <= 0.0 {
            return Err(ArgminError::InvalidParameter {
                parameter: "HagerZhangLineSearch: theta must be > 0.0.".to_string(),
            }
            .into());
        }
        if theta >= 1.0 {
            return Err(ArgminError::InvalidParameter {
                parameter: "HagerZhangLineSearch: theta must be < 1.0.".to_string(),
            }
            .into());
        }
        self.theta = theta;
        Ok(self)
    }

    /// set gamma
    pub fn set_gamma(&mut self, gamma: f64) -> Result<&mut Self, Error> {
        if gamma <= 0.0 {
            return Err(ArgminError::InvalidParameter {
                parameter: "HagerZhangLineSearch: gamma must be > 0.0.".to_string(),
            }
            .into());
        }
        if gamma >= 1.0 {
            return Err(ArgminError::InvalidParameter {
                parameter: "HagerZhangLineSearch: gamma must be < 1.0.".to_string(),
            }
            .into());
        }
        self.gamma = gamma;
        Ok(self)
    }

    /// set eta
    pub fn set_eta(&mut self, eta: f64) -> Result<&mut Self, Error> {
        if eta <= 0.0 {
            return Err(ArgminError::InvalidParameter {
                parameter: "HagerZhangLineSearch: eta must be > 0.0.".to_string(),
            }
            .into());
        }
        self.eta = eta;
        Ok(self)
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
        if self.sigma < self.delta {
            return Err(ArgminError::InvalidParameter {
                parameter: "HagerZhangLineSearch: sigma must be >= delta.".to_string(),
            }
            .into());
        }
        unimplemented!()
    }

    fn next_iter(&mut self) -> Result<ArgminIterationData<Self::Parameters>, Error> {
        // let out = ArgminIterationData::new(new_param, self.stp.fx);
        // Ok(out)
        unimplemented!()
    }
}
