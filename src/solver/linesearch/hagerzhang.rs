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
///
/// Parameters for interval:
///   a_x, a_f, a_g, b_x, b_f, b_g
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
    /// epsilon_k
    epsilon_k: f64,
    /// theta: (0, 1), used in the update rules when the potential intervals [a, c] or [c, b]
    /// viloate the opposite slope condition
    theta: f64,
    /// gamma: (0, 1), determines when a bisection step is performed
    gamma: f64,
    /// eta: (0, infinity), used in the lower bound for beta_k^N
    eta: f64,
    /// a
    a_x: f64,
    /// phi(a)
    a_f: f64,
    /// phi'(a)
    a_g: f64,
    /// b
    b_x: f64,
    /// phi(b)
    b_f: f64,
    /// phi'(b)
    b_g: f64,
    /// c
    c_x: f64,
    /// phi(c)
    c_f: f64,
    /// phi'(c)
    c_g: f64,
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
    /// initial gradient (builder)
    init_grad: T,
    /// Search direction (builder)
    search_direction: T,
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
            epsilon_k: std::f64::NAN,
            theta: 0.5,
            gamma: 0.66,
            eta: 0.01,
            a_x: std::f64::NAN,
            a_f: std::f64::NAN,
            a_g: std::f64::NAN,
            b_x: std::f64::NAN,
            b_f: std::f64::NAN,
            b_g: std::f64::NAN,
            c_x: std::f64::NAN,
            c_f: std::f64::NAN,
            c_g: std::f64::NAN,
            init_param_b: None,
            finit_b: None,
            init_grad_b: None,
            search_direction_b: None,
            init_param: T::default(),
            init_grad: T::default(),
            search_direction: T::default(),
            finit: std::f64::INFINITY,
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

    fn update(
        &mut self,
        (a_x, a_f, a_g): (f64, f64, f64),
        (b_x, b_f, b_g): (f64, f64, f64),
        (c_x, c_f, c_g): (f64, f64, f64),
    ) -> Result<((f64, f64, f64), (f64, f64, f64)), Error> {
        // U0
        if c_x <= a_x || c_x >= b_x {
            // nothing changes.
            return Ok(((a_x, a_f, a_g), (b_x, b_f, b_g)));
        }

        // U1
        if c_g >= 0.0 {
            return Ok(((a_x, a_f, a_g), (c_x, c_f, c_g)));
        }

        // U2
        if c_g < 0.0 && c_g <= self.finit + self.epsilon_k {
            return Ok(((c_x, c_f, c_g), (b_x, b_f, b_g)));
        }

        // U3
        if c_g < 0.0 && c_g > self.finit + self.epsilon_k {
            let mut ah_x = a_x;
            let mut ah_f = a_f;
            let mut ah_g = a_g;
            let mut bh_x = c_x;
            loop {
                let d_x = (1.0 - self.theta) * ah_x + self.theta * bh_x;
                let tmp = self
                    .init_param
                    .scaled_add(d_x, self.search_direction.clone());
                let d_f = self.apply(&tmp)?;
                let grad = self.gradient(&tmp)?;
                let d_g = self.search_direction.dot(grad);
                if d_g >= 0.0 {
                    return Ok(((ah_x, ah_f, ah_g), (d_x, d_f, d_g)));
                }
                if d_g < 0.0 && d_f <= self.finit + self.epsilon_k {
                    ah_x = d_x;
                    ah_f = d_f;
                    ah_g = d_g;
                }
                if d_g < 0.0 && d_f > self.finit + self.epsilon_k {
                    bh_x = d_x;
                }
            }
        }

        return Err(ArgminError::InvalidParameter {
            parameter: "HagerZhangLineSearch: Reached unreachable point in `update` method."
                .to_string(),
        }
        .into());
    }

    /// secant step
    fn secant(&self, a_x: f64, a_g: f64, b_x: f64, b_g: f64) -> f64 {
        (a_x * b_g - b_x * a_g) / (b_g - a_g)
    }

    /// double secant step
    fn secant2(
        &mut self,
        (a_x, a_f, a_g): (f64, f64, f64),
        (b_x, b_f, b_g): (f64, f64, f64),
    ) -> Result<((f64, f64, f64), (f64, f64, f64)), Error> {
        // S1
        let c_x = self.secant(a_x, a_g, b_x, b_g);
        let tmp = self
            .init_param
            .scaled_add(c_x, self.search_direction.clone());
        let c_f = self.apply(&tmp)?;
        let grad = self.gradient(&tmp)?;
        let c_g = self.search_direction.dot(grad);

        let ((aa_x, aa_f, aa_g), (bb_x, bb_f, bb_g)) =
            self.update((a_x, a_f, a_g), (b_x, b_f, b_g), (c_x, c_f, c_g))?;

        let mut c_bar_x: f64 = 0.0;

        // S2
        if c_x == bb_x {
            c_bar_x = self.secant(b_x, b_g, bb_x, bb_g);
        }

        // S3
        if c_x == aa_x {
            c_bar_x = self.secant(a_x, a_g, aa_x, aa_g);
        }

        // S4
        if c_x == aa_x || c_x == bb_x {
            let tmp = self
                .init_param
                .scaled_add(c_bar_x, self.search_direction.clone());
            let c_bar_f = self.apply(&tmp)?;
            let grad = self.gradient(&tmp)?;
            let c_bar_g = self.search_direction.dot(grad);

            let (a_bar, b_bar) = self.update(
                (aa_x, aa_f, aa_g),
                (bb_x, bb_f, bb_g),
                (c_bar_x, c_bar_f, c_bar_g),
            )?;
            return Ok((a_bar, b_bar));
        } else {
            return Ok(((aa_x, aa_f, aa_g), (bb_x, bb_f, bb_g)));
        }

        return Err(ArgminError::InvalidParameter {
            parameter: "HagerZhangLineSearch: Reached unreachable point in `secant` method."
                .to_string(),
        }
        .into());
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
        self.search_direction_b = Some(search_direction);
    }

    /// Set initial parameter
    fn set_initial_parameter(&mut self, param: T) {
        self.init_param_b = Some(param.clone());
        self.base.set_cur_param(param);
    }

    /// Set initial cost function value
    fn set_initial_cost(&mut self, init_cost: f64) {
        self.finit_b = Some(init_cost);
    }

    /// Set initial gradient
    fn set_initial_gradient(&mut self, init_grad: T) {
        self.init_grad_b = Some(init_grad);
    }

    /// Calculate initial cost function value
    fn calc_initial_cost(&mut self) -> Result<(), Error> {
        let tmp = self.base.cur_param();
        self.finit_b = Some(self.apply(&tmp)?);
        Ok(())
    }

    /// Calculate initial cost function value
    fn calc_initial_gradient(&mut self) -> Result<(), Error> {
        let tmp = self.base.cur_param();
        self.init_grad_b = Some(self.gradient(&tmp)?);
        Ok(())
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

        self.init_param = check_param!(
            self.init_param_b,
            "HagerZhangLineSearch: Initial parameter not initialized. Call `set_initial_parameter`."
        );

        self.finit = check_param!(
            self.finit_b,
            "HagerZhangLineSearch: Initial cost not computed. Call `set_initial_cost` or `calc_inital_cost`."
        );

        self.init_grad = check_param!(
            self.init_grad_b,
            "HagerZhangLineSearch: Initial gradient not computed. Call `set_initial_grad` or `calc_inital_grad`."
        );

        self.search_direction = check_param!(
            self.search_direction_b,
            "HagerZhangLineSearch: Search direction not initialized. Call `set_search_direction`."
        );

        Ok(())
    }

    fn next_iter(&mut self) -> Result<ArgminIterationData<Self::Parameters>, Error> {
        // let out = ArgminIterationData::new(new_param, self.stp.fx);
        // Ok(out)
        unimplemented!()
    }
}
