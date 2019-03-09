// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! * [Hager-Zhang line search](struct.HagerZhangLineSearch.html)
//!
//! TODO: Not all stopping criteria implemented
//!
//! # Reference
//!
//! William W. Hager and Hongchao Zhang. "A new conjugate gradient method with guaranteed descent
//! and an efficient line search." SIAM J. Optim. 16(1), 2006, 170-192.
//! DOI: https://doi.org/10.1137/030601880

use crate::prelude::*;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::default::Default;

type Triplet = (f64, f64, f64);

/// The Hager-Zhang line search is a method to find a step length which obeys the strong Wolfe
/// conditions.
///
/// # Example
///
/// ```
/// # extern crate argmin;
/// # use argmin::prelude::*;
/// # use argmin::solver::linesearch::HagerZhangLineSearch;
/// # use argmin::testfunctions::{sphere, sphere_derivative};
/// # use serde::{Deserialize, Serialize};
/// #
/// # #[derive(Clone, Default, Serialize, Deserialize)]
/// # struct MyProblem {}
/// #
/// # impl ArgminOp for MyProblem {
/// #     type Param = Vec<f64>;
/// #     type Output = f64;
/// #     type Hessian = ();
/// #
/// #     fn apply(&self, param: &Vec<f64>) -> Result<f64, Error> {
/// #         Ok(sphere(param))
/// #     }
/// #
/// #     fn gradient(&self, param: &Vec<f64>) -> Result<Vec<f64>, Error> {
/// #         Ok(sphere_derivative(param))
/// #     }
/// # }
/// #
/// # fn run() -> Result<(), Error> {
/// // Define inital parameter vector
/// let init_param: Vec<f64> = vec![1.0, 0.0];
///
/// // Problem definition
/// let operator = MyProblem {};
///
/// // Set up line search method
/// let mut solver = HagerZhangLineSearch::new(operator);
///
/// // Set search direction
/// solver.set_search_direction(vec![-2.0, 0.0]);
///
/// // Set initial position
/// solver.set_initial_parameter(init_param);
///
/// // Calculate initial cost ...
/// solver.calc_initial_cost()?;
/// // ... or, alternatively, set cost if it is already computed
/// // solver.set_initial_cost(...);
///
/// // Calculate initial gradient ...
/// solver.calc_initial_gradient()?;
/// // .. or, alternatively, set gradient if it is already computed
/// // solver.set_initial_gradient(...);
///
/// // Set initial step length
/// solver.set_initial_alpha(1.0)?;
///
/// // Attach a logger
/// solver.add_logger(ArgminSlogLogger::term());
///
/// // Run solver
/// solver.run()?;
///
/// // Wait a second (lets the logger flush everything before printing again)
/// std::thread::sleep(std::time::Duration::from_secs(1));
///
/// // Print Result
/// println!("{:?}", solver.result());
/// #     Ok(())
/// # }
/// #
/// # fn main() {
/// #     if let Err(ref e) = run() {
/// #         println!("{} {}", e.as_fail(), e.backtrace());
/// #     }
/// # }
/// ```
///
/// # References
///
/// [0] William W. Hager and Hongchao Zhang. "A new conjugate gradient method with guaranteed
/// descent and an efficient line search." SIAM J. Optim. 16(1), 2006, 170-192.
/// DOI: https://doi.org/10.1137/030601880
#[derive(Serialize, Deserialize, Clone)]
pub struct HagerZhangLineSearch<P> {
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
    /// initial a
    a_x_init: f64,
    /// a
    a_x: f64,
    /// phi(a)
    a_f: f64,
    /// phi'(a)
    a_g: f64,
    /// initial b
    b_x_init: f64,
    /// b
    b_x: f64,
    /// phi(b)
    b_f: f64,
    /// phi'(b)
    b_g: f64,
    /// initial c
    c_x_init: f64,
    /// c
    c_x: f64,
    /// phi(c)
    c_f: f64,
    /// phi'(c)
    c_g: f64,
    /// best x
    best_x: f64,
    /// best function value
    best_f: f64,
    /// best slope
    best_g: f64,
    /// Search direction (builder)
    search_direction_b: Option<P>,
    /// initial parameter vector
    init_param: P,
    /// initial cost
    finit: f64,
    /// initial gradient (builder)
    init_grad: P,
    /// Search direction (builder)
    search_direction: P,
    /// Search direction in 1D
    dginit: f64,
}

impl<P> HagerZhangLineSearch<P>
where
    P: Clone
        + Default
        + Serialize
        + DeserializeOwned
        + ArgminScaledAdd<P, f64, P>
        + ArgminDot<P, f64>,
{
    /// Constructor
    ///
    /// Parameters:
    ///
    /// `operator`: operator
    pub fn new() -> Self {
        HagerZhangLineSearch {
            delta: 0.1,
            sigma: 0.9,
            epsilon: 1e-6,
            epsilon_k: std::f64::NAN,
            theta: 0.5,
            gamma: 0.66,
            eta: 0.01,
            a_x_init: std::f64::EPSILON,
            a_x: std::f64::NAN,
            a_f: std::f64::NAN,
            a_g: std::f64::NAN,
            b_x_init: 100.0,
            b_x: std::f64::NAN,
            b_f: std::f64::NAN,
            b_g: std::f64::NAN,
            c_x_init: 1.0,
            c_x: std::f64::NAN,
            c_f: std::f64::NAN,
            c_g: std::f64::NAN,
            best_x: 0.0,
            best_f: std::f64::INFINITY,
            best_g: std::f64::NAN,
            search_direction_b: None,
            init_param: P::default(),
            init_grad: P::default(),
            search_direction: P::default(),
            dginit: std::f64::NAN,
            finit: std::f64::INFINITY,
        }
    }

    /// set delta
    pub fn delta(mut self, delta: f64) -> Result<Self, Error> {
        if delta <= 0.0 {
            return Err(ArgminError::InvalidParameter {
                text: "HagerZhangLineSearch: delta must be > 0.0.".to_string(),
            }
            .into());
        }
        if delta >= 1.0 {
            return Err(ArgminError::InvalidParameter {
                text: "HagerZhangLineSearch: delta must be < 1.0.".to_string(),
            }
            .into());
        }
        self.delta = delta;
        Ok(self)
    }

    /// set sigma
    pub fn sigma(mut self, sigma: f64) -> Result<Self, Error> {
        if sigma < self.delta {
            return Err(ArgminError::InvalidParameter {
                text: "HagerZhangLineSearch: sigma must be >= delta.".to_string(),
            }
            .into());
        }
        if sigma >= 1.0 {
            return Err(ArgminError::InvalidParameter {
                text: "HagerZhangLineSearch: sigma must be < 1.0.".to_string(),
            }
            .into());
        }
        self.sigma = sigma;
        Ok(self)
    }

    /// set epsilon
    pub fn epsilon(mut self, epsilon: f64) -> Result<Self, Error> {
        if epsilon < 0.0 {
            return Err(ArgminError::InvalidParameter {
                text: "HagerZhangLineSearch: epsilon must be >= 0.0.".to_string(),
            }
            .into());
        }
        self.epsilon = epsilon;
        Ok(self)
    }

    /// set theta
    pub fn theta(mut self, theta: f64) -> Result<Self, Error> {
        if theta <= 0.0 {
            return Err(ArgminError::InvalidParameter {
                text: "HagerZhangLineSearch: theta must be > 0.0.".to_string(),
            }
            .into());
        }
        if theta >= 1.0 {
            return Err(ArgminError::InvalidParameter {
                text: "HagerZhangLineSearch: theta must be < 1.0.".to_string(),
            }
            .into());
        }
        self.theta = theta;
        Ok(self)
    }

    /// set gamma
    pub fn gamma(mut self, gamma: f64) -> Result<Self, Error> {
        if gamma <= 0.0 {
            return Err(ArgminError::InvalidParameter {
                text: "HagerZhangLineSearch: gamma must be > 0.0.".to_string(),
            }
            .into());
        }
        if gamma >= 1.0 {
            return Err(ArgminError::InvalidParameter {
                text: "HagerZhangLineSearch: gamma must be < 1.0.".to_string(),
            }
            .into());
        }
        self.gamma = gamma;
        Ok(self)
    }

    /// set eta
    pub fn eta(mut self, eta: f64) -> Result<Self, Error> {
        if eta <= 0.0 {
            return Err(ArgminError::InvalidParameter {
                text: "HagerZhangLineSearch: eta must be > 0.0.".to_string(),
            }
            .into());
        }
        self.eta = eta;
        Ok(self)
    }

    /// set alpha limits
    pub fn alpha(mut self, alpha_min: f64, alpha_max: f64) -> Result<Self, Error> {
        if alpha_min < 0.0 {
            return Err(ArgminError::InvalidParameter {
                text: "HagerZhangLineSearch: alpha_min must be >= 0.0.".to_string(),
            }
            .into());
        }
        if alpha_max <= alpha_min {
            return Err(ArgminError::InvalidParameter {
                text: "HagerZhangLineSearch: alpha_min must be smaller than alpha_max.".to_string(),
            }
            .into());
        }
        self.a_x_init = alpha_min;
        self.b_x_init = alpha_max;
        Ok(self)
    }

    fn update<O: ArgminOp<Param = P, Output = f64>>(
        &mut self,
        op: &mut OpWrapper<O>,
        (a_x, a_f, a_g): Triplet,
        (b_x, b_f, b_g): Triplet,
        (c_x, c_f, c_g): Triplet,
    ) -> Result<(Triplet, Triplet), Error> {
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
        if c_g < 0.0 && c_f <= self.finit + self.epsilon_k {
            return Ok(((c_x, c_f, c_g), (b_x, b_f, b_g)));
        }

        // U3
        if c_g < 0.0 && c_f > self.finit + self.epsilon_k {
            let mut ah_x = a_x;
            let mut ah_f = a_f;
            let mut ah_g = a_g;
            let mut bh_x = c_x;
            loop {
                let d_x = (1.0 - self.theta) * ah_x + self.theta * bh_x;
                let d_f = self.calc(op, d_x)?;
                let d_g = self.calc_grad(op, d_x)?;
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

        // return Ok(((a_x, a_f, a_g), (b_x, b_f, b_g)));
        Err(ArgminError::InvalidParameter {
            text: "HagerZhangLineSearch: Reached unreachable point in `update` method.".to_string(),
        }
        .into())
    }

    /// secant step
    fn secant(&self, a_x: f64, a_g: f64, b_x: f64, b_g: f64) -> f64 {
        (a_x * b_g - b_x * a_g) / (b_g - a_g)
    }

    /// double secant step
    fn secant2<O: ArgminOp<Param = P, Output = f64>>(
        &mut self,
        op: &mut OpWrapper<O>,
        (a_x, a_f, a_g): Triplet,
        (b_x, b_f, b_g): Triplet,
    ) -> Result<(Triplet, Triplet), Error> {
        // S1
        let c_x = self.secant(a_x, a_g, b_x, b_g);
        let c_f = self.calc(op, c_x)?;
        let c_g = self.calc_grad(op, c_x)?;
        let mut c_bar_x: f64 = 0.0;

        let ((aa_x, aa_f, aa_g), (bb_x, bb_f, bb_g)) =
            self.update(op, (a_x, a_f, a_g), (b_x, b_f, b_g), (c_x, c_f, c_g))?;

        // S2
        if (c_x - bb_x).abs() < std::f64::EPSILON {
            c_bar_x = self.secant(b_x, b_g, bb_x, bb_g);
        }

        // S3
        if (c_x - aa_x).abs() < std::f64::EPSILON {
            c_bar_x = self.secant(a_x, a_g, aa_x, aa_g);
        }

        // S4
        if (c_x - aa_x).abs() < std::f64::EPSILON || (c_x - bb_x).abs() < std::f64::EPSILON {
            let c_bar_f = self.calc(op, c_bar_x)?;
            let c_bar_g = self.calc_grad(op, c_bar_x)?;

            let (a_bar, b_bar) = self.update(
                op,
                (aa_x, aa_f, aa_g),
                (bb_x, bb_f, bb_g),
                (c_bar_x, c_bar_f, c_bar_g),
            )?;
            Ok((a_bar, b_bar))
        } else {
            Ok(((aa_x, aa_f, aa_g), (bb_x, bb_f, bb_g)))
        }
    }

    fn calc<O: ArgminOp<Param = P, Output = f64>>(
        &mut self,
        op: &mut OpWrapper<O>,
        alpha: f64,
    ) -> Result<f64, Error> {
        let tmp = self.init_param.scaled_add(&alpha, &self.search_direction);
        op.apply(&tmp)
    }

    fn calc_grad<O: ArgminOp<Param = P, Output = f64>>(
        &mut self,
        op: &mut OpWrapper<O>,
        alpha: f64,
    ) -> Result<f64, Error> {
        let tmp = self.init_param.scaled_add(&alpha, &self.search_direction);
        let grad = op.gradient(&tmp)?;
        Ok(self.search_direction.dot(&grad))
    }

    fn set_best(&mut self) {
        if self.a_f < self.b_f && self.a_f < self.c_f {
            self.best_x = self.a_x;
            self.best_f = self.a_f;
            self.best_g = self.a_g;
        }

        if self.b_f < self.a_f && self.b_f < self.c_f {
            self.best_x = self.b_x;
            self.best_f = self.b_f;
            self.best_g = self.b_g;
        }

        if self.c_f < self.a_f && self.c_f < self.b_f {
            self.best_x = self.c_x;
            self.best_f = self.c_f;
            self.best_g = self.c_g;
        }
    }
}

impl<P> ArgminLineSearch<P> for HagerZhangLineSearch<P>
where
    P: Clone
        + Default
        + Serialize
        + ArgminSub<P, P>
        + ArgminDot<P, f64>
        + ArgminScaledAdd<P, f64, P>,
{
    /// Set search direction
    fn set_search_direction(&mut self, search_direction: P) {
        self.search_direction_b = Some(search_direction);
    }

    /// Set initial alpha value
    fn set_init_alpha(&mut self, alpha: f64) -> Result<(), Error> {
        self.c_x_init = alpha;
        Ok(())
    }
}

impl<P, O> Solver<O> for HagerZhangLineSearch<P>
where
    O: ArgminOp<Param = P, Output = f64>,
    P: Clone
        + Default
        + Serialize
        + DeserializeOwned
        + ArgminSub<P, P>
        + ArgminDot<P, f64>
        + ArgminScaledAdd<P, f64, P>,
{
    fn init(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &IterState<O>,
    ) -> Result<Option<ArgminIterData<O>>, Error> {
        if self.sigma < self.delta {
            return Err(ArgminError::InvalidParameter {
                text: "HagerZhangLineSearch: sigma must be >= delta.".to_string(),
            }
            .into());
        }

        self.search_direction = check_param!(
            self.search_direction_b,
            "HagerZhangLineSearch: Search direction not initialized. Call `set_search_direction`."
        );

        self.init_param = state.get_param();

        let cost = state.get_cost();
        self.finit = if cost == std::f64::INFINITY {
            op.apply(&self.init_param)?
        } else {
            cost
        };

        self.init_grad = state.get_grad().unwrap_or(op.gradient(&self.init_param)?);

        self.a_x = self.a_x_init;
        self.b_x = self.b_x_init;
        self.c_x = self.c_x_init;

        let at = self.a_x;
        self.a_f = self.calc(op, at)?;
        self.a_g = self.calc_grad(op, at)?;
        let bt = self.b_x;
        self.b_f = self.calc(op, bt)?;
        self.b_g = self.calc_grad(op, bt)?;
        let ct = self.c_x;
        self.c_f = self.calc(op, ct)?;
        self.c_g = self.calc_grad(op, ct)?;

        self.epsilon_k = self.epsilon * self.finit.abs();

        self.dginit = self.init_grad.dot(&self.search_direction);

        self.set_best();
        let new_param = self
            .init_param
            .scaled_add(&self.best_x, &self.search_direction);
        let best_f = self.best_f;

        Ok(Some(ArgminIterData::new().param(new_param).cost(best_f)))
    }

    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        _state: &IterState<O>,
    ) -> Result<ArgminIterData<O>, Error> {
        // L1
        let aa = (self.a_x, self.a_f, self.a_g);
        let bb = (self.b_x, self.b_f, self.b_g);
        let ((mut at_x, mut at_f, mut at_g), (mut bt_x, mut bt_f, mut bt_g)) =
            self.secant2(op, aa, bb)?;

        // L2
        if bt_x - at_x > self.gamma * (self.b_x - self.a_x) {
            let c_x = (at_x + bt_x) / 2.0;
            let tmp = self.init_param.scaled_add(&c_x, &self.search_direction);
            let c_f = op.apply(&tmp)?;
            let grad = op.gradient(&tmp)?;
            let c_g = self.search_direction.dot(&grad);
            let ((an_x, an_f, an_g), (bn_x, bn_f, bn_g)) =
                self.update(op, (at_x, at_f, at_g), (bt_x, bt_f, bt_g), (c_x, c_f, c_g))?;
            at_x = an_x;
            at_f = an_f;
            at_g = an_g;
            bt_x = bn_x;
            bt_f = bn_f;
            bt_g = bn_g;
        }

        // L3
        self.a_x = at_x;
        self.a_f = at_f;
        self.a_g = at_g;
        self.b_x = bt_x;
        self.b_f = bt_f;
        self.b_g = bt_g;

        self.set_best();
        let new_param = self
            .init_param
            .scaled_add(&self.best_x, &self.search_direction);
        Ok(ArgminIterData::new().param(new_param).cost(self.best_f))
    }

    fn terminate(&mut self, _state: &IterState<O>) -> TerminationReason {
        if self.best_f - self.finit < self.delta * self.best_x * self.dginit {
            return TerminationReason::LineSearchConditionMet;
        }
        if self.best_g > self.sigma * self.dginit {
            return TerminationReason::LineSearchConditionMet;
        }
        if (2.0 * self.delta - 1.0) * self.dginit >= self.best_g
            && self.best_g >= self.sigma * self.dginit
            && self.best_f <= self.finit + self.epsilon_k
        {
            return TerminationReason::LineSearchConditionMet;
        }
        TerminationReason::NotTerminated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::send_sync_test;
    use crate::MinimalNoOperator;

    send_sync_test!(hagerzhang, HagerZhangLineSearch<MinimalNoOperator>);
}
