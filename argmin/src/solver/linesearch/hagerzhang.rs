// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! * [Hager-Zhang line search](struct.HagerZhangLineSearch.html)
//!
//! # Reference
//!
//! William W. Hager and Hongchao Zhang. "A new conjugate gradient method with guaranteed descent
//! and an efficient line search." SIAM J. Optim. 16(1), 2006, 170-192.
//! DOI: <https://doi.org/10.1137/030601880>

use crate::core::{
    ArgminError, ArgminFloat, ArgminIterData, ArgminLineSearch, ArgminOp, Error, IterState,
    OpWrapper, SerializeAlias, Solver, TerminationReason,
};
use argmin_math::{ArgminDot, ArgminScaledAdd};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

type Triplet<F> = (F, F, F);

/// The Hager-Zhang line search is a method to find a step length which obeys the strong Wolfe
/// conditions.
///
/// # References
///
/// \[0\] William W. Hager and Hongchao Zhang. "A new conjugate gradient method with guaranteed
/// descent and an efficient line search." SIAM J. Optim. 16(1), 2006, 170-192.
/// DOI: <https://doi.org/10.1137/030601880>
#[derive(Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct HagerZhangLineSearch<P, F> {
    /// delta: (0, 0.5), used in the Wolfe conditions
    delta: F,
    /// sigma: [delta, 1), used in the Wolfe conditions
    sigma: F,
    /// epsilon: [0, infinity), used in the approximate Wolfe termination
    epsilon: F,
    /// epsilon_k
    epsilon_k: F,
    /// theta: (0, 1), used in the update rules when the potential intervals [a, c] or [c, b]
    /// violate the opposite slope condition
    theta: F,
    /// gamma: (0, 1), determines when a bisection step is performed
    gamma: F,
    /// eta: (0, infinity), used in the lower bound for beta_k^N
    eta: F,
    /// initial a
    a_x_init: F,
    /// a
    a_x: F,
    /// phi(a)
    a_f: F,
    /// phi'(a)
    a_g: F,
    /// initial b
    b_x_init: F,
    /// b
    b_x: F,
    /// phi(b)
    b_f: F,
    /// phi'(b)
    b_g: F,
    /// initial c
    c_x_init: F,
    /// c
    c_x: F,
    /// phi(c)
    c_f: F,
    /// phi'(c)
    c_g: F,
    /// best x
    best_x: F,
    /// best function value
    best_f: F,
    /// best slope
    best_g: F,
    /// initial parameter vector
    init_param: Option<P>,
    /// initial cost
    finit: F,
    /// initial gradient (builder)
    init_grad: Option<P>,
    /// Search direction (builder)
    search_direction: Option<P>,
    /// Search direction in 1D
    dginit: F,
}

impl<P, F> HagerZhangLineSearch<P, F>
where
    F: ArgminFloat,
{
    /// Constructor
    pub fn new() -> Self {
        HagerZhangLineSearch {
            delta: F::from_f64(0.1).unwrap(),
            sigma: F::from_f64(0.9).unwrap(),
            epsilon: F::from_f64(1e-6).unwrap(),
            epsilon_k: F::nan(),
            theta: F::from_f64(0.5).unwrap(),
            gamma: F::from_f64(0.66).unwrap(),
            eta: F::from_f64(0.01).unwrap(),
            a_x_init: F::epsilon(),
            a_x: F::nan(),
            a_f: F::nan(),
            a_g: F::nan(),
            b_x_init: F::from_f64(100.0).unwrap(),
            b_x: F::nan(),
            b_f: F::nan(),
            b_g: F::nan(),
            c_x_init: F::from_f64(1.0).unwrap(),
            c_x: F::nan(),
            c_f: F::nan(),
            c_g: F::nan(),
            best_x: F::from_f64(0.0).unwrap(),
            best_f: F::infinity(),
            best_g: F::nan(),
            init_param: None,
            init_grad: None,
            search_direction: None,
            dginit: F::nan(),
            finit: F::infinity(),
        }
    }
}

impl<P, F> HagerZhangLineSearch<P, F>
where
    P: ArgminScaledAdd<P, F, P> + ArgminDot<P, F>,
    F: ArgminFloat,
{
    /// set delta
    pub fn delta(mut self, delta: F) -> Result<Self, Error> {
        if delta <= F::from_f64(0.0).unwrap() {
            return Err(ArgminError::InvalidParameter {
                text: "HagerZhangLineSearch: delta must be > 0.0.".to_string(),
            }
            .into());
        }
        if delta >= F::from_f64(1.0).unwrap() {
            return Err(ArgminError::InvalidParameter {
                text: "HagerZhangLineSearch: delta must be < 1.0.".to_string(),
            }
            .into());
        }
        self.delta = delta;
        Ok(self)
    }

    /// set sigma
    pub fn sigma(mut self, sigma: F) -> Result<Self, Error> {
        if sigma < self.delta {
            return Err(ArgminError::InvalidParameter {
                text: "HagerZhangLineSearch: sigma must be >= delta.".to_string(),
            }
            .into());
        }
        if sigma >= F::from_f64(1.0).unwrap() {
            return Err(ArgminError::InvalidParameter {
                text: "HagerZhangLineSearch: sigma must be < 1.0.".to_string(),
            }
            .into());
        }
        self.sigma = sigma;
        Ok(self)
    }

    /// set epsilon
    pub fn epsilon(mut self, epsilon: F) -> Result<Self, Error> {
        if epsilon < F::from_f64(0.0).unwrap() {
            return Err(ArgminError::InvalidParameter {
                text: "HagerZhangLineSearch: epsilon must be >= 0.0.".to_string(),
            }
            .into());
        }
        self.epsilon = epsilon;
        Ok(self)
    }

    /// set theta
    pub fn theta(mut self, theta: F) -> Result<Self, Error> {
        if theta <= F::from_f64(0.0).unwrap() {
            return Err(ArgminError::InvalidParameter {
                text: "HagerZhangLineSearch: theta must be > 0.0.".to_string(),
            }
            .into());
        }
        if theta >= F::from_f64(1.0).unwrap() {
            return Err(ArgminError::InvalidParameter {
                text: "HagerZhangLineSearch: theta must be < 1.0.".to_string(),
            }
            .into());
        }
        self.theta = theta;
        Ok(self)
    }

    /// set gamma
    pub fn gamma(mut self, gamma: F) -> Result<Self, Error> {
        if gamma <= F::from_f64(0.0).unwrap() {
            return Err(ArgminError::InvalidParameter {
                text: "HagerZhangLineSearch: gamma must be > 0.0.".to_string(),
            }
            .into());
        }
        if gamma >= F::from_f64(1.0).unwrap() {
            return Err(ArgminError::InvalidParameter {
                text: "HagerZhangLineSearch: gamma must be < 1.0.".to_string(),
            }
            .into());
        }
        self.gamma = gamma;
        Ok(self)
    }

    /// set eta
    pub fn eta(mut self, eta: F) -> Result<Self, Error> {
        if eta <= F::from_f64(0.0).unwrap() {
            return Err(ArgminError::InvalidParameter {
                text: "HagerZhangLineSearch: eta must be > 0.0.".to_string(),
            }
            .into());
        }
        self.eta = eta;
        Ok(self)
    }

    /// set alpha limits
    pub fn alpha(mut self, alpha_min: F, alpha_max: F) -> Result<Self, Error> {
        if alpha_min < F::from_f64(0.0).unwrap() {
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

    fn update<O: ArgminOp<Param = P, Output = F>>(
        &mut self,
        op: &mut OpWrapper<O>,
        (a_x, a_f, a_g): Triplet<F>,
        (b_x, b_f, b_g): Triplet<F>,
        (c_x, c_f, c_g): Triplet<F>,
    ) -> Result<(Triplet<F>, Triplet<F>), Error> {
        // U0
        if c_x <= a_x || c_x >= b_x {
            // nothing changes.
            return Ok(((a_x, a_f, a_g), (b_x, b_f, b_g)));
        }

        // U1
        if c_g >= F::from_f64(0.0).unwrap() {
            return Ok(((a_x, a_f, a_g), (c_x, c_f, c_g)));
        }

        // U2
        if c_g < F::from_f64(0.0).unwrap() && c_f <= self.finit + self.epsilon_k {
            return Ok(((c_x, c_f, c_g), (b_x, b_f, b_g)));
        }

        // U3
        if c_g < F::from_f64(0.0).unwrap() && c_f > self.finit + self.epsilon_k {
            let mut ah_x = a_x;
            let mut ah_f = a_f;
            let mut ah_g = a_g;
            let mut bh_x = c_x;
            loop {
                let d_x = (F::from_f64(1.0).unwrap() - self.theta) * ah_x + self.theta * bh_x;
                let d_f = self.calc(op, d_x)?;
                let d_g = self.calc_grad(op, d_x)?;
                if d_g >= F::from_f64(0.0).unwrap() {
                    return Ok(((ah_x, ah_f, ah_g), (d_x, d_f, d_g)));
                }
                if d_g < F::from_f64(0.0).unwrap() && d_f <= self.finit + self.epsilon_k {
                    ah_x = d_x;
                    ah_f = d_f;
                    ah_g = d_g;
                }
                if d_g < F::from_f64(0.0).unwrap() && d_f > self.finit + self.epsilon_k {
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
    fn secant(&self, a_x: F, a_g: F, b_x: F, b_g: F) -> F {
        (a_x * b_g - b_x * a_g) / (b_g - a_g)
    }

    /// double secant step
    fn secant2<O: ArgminOp<Param = P, Output = F>>(
        &mut self,
        op: &mut OpWrapper<O>,
        (a_x, a_f, a_g): Triplet<F>,
        (b_x, b_f, b_g): Triplet<F>,
    ) -> Result<(Triplet<F>, Triplet<F>), Error> {
        // S1
        let c_x = self.secant(a_x, a_g, b_x, b_g);
        let c_f = self.calc(op, c_x)?;
        let c_g = self.calc_grad(op, c_x)?;
        let mut c_bar_x: F = F::from_f64(0.0).unwrap();

        let ((aa_x, aa_f, aa_g), (bb_x, bb_f, bb_g)) =
            self.update(op, (a_x, a_f, a_g), (b_x, b_f, b_g), (c_x, c_f, c_g))?;

        // S2
        if (c_x - bb_x).abs() < F::epsilon() {
            c_bar_x = self.secant(b_x, b_g, bb_x, bb_g);
        }

        // S3
        if (c_x - aa_x).abs() < F::epsilon() {
            c_bar_x = self.secant(a_x, a_g, aa_x, aa_g);
        }

        // S4
        if (c_x - aa_x).abs() < F::epsilon() || (c_x - bb_x).abs() < F::epsilon() {
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

    fn calc<O: ArgminOp<Param = P, Output = F>>(
        &mut self,
        op: &mut OpWrapper<O>,
        alpha: F,
    ) -> Result<F, Error> {
        let tmp = self
            .init_param
            .as_ref()
            .unwrap()
            .scaled_add(&alpha, self.search_direction.as_ref().unwrap());
        op.apply(&tmp)
    }

    fn calc_grad<O: ArgminOp<Param = P, Output = F>>(
        &mut self,
        op: &mut OpWrapper<O>,
        alpha: F,
    ) -> Result<F, Error> {
        let tmp = self
            .init_param
            .as_ref()
            .unwrap()
            .scaled_add(&alpha, self.search_direction.as_ref().unwrap());
        let grad = op.gradient(&tmp)?;
        Ok(self.search_direction.as_ref().unwrap().dot(&grad))
    }

    fn set_best(&mut self) {
        if self.a_f <= self.b_f && self.a_f <= self.c_f {
            self.best_x = self.a_x;
            self.best_f = self.a_f;
            self.best_g = self.a_g;
        }

        if self.b_f <= self.a_f && self.b_f <= self.c_f {
            self.best_x = self.b_x;
            self.best_f = self.b_f;
            self.best_g = self.b_g;
        }

        if self.c_f <= self.a_f && self.c_f <= self.b_f {
            self.best_x = self.c_x;
            self.best_f = self.c_f;
            self.best_g = self.c_g;
        }
    }
}

impl<P, F> Default for HagerZhangLineSearch<P, F>
where
    F: ArgminFloat,
{
    fn default() -> Self {
        HagerZhangLineSearch::new()
    }
}

impl<P, F> ArgminLineSearch<P, F> for HagerZhangLineSearch<P, F> {
    /// Set search direction
    fn set_search_direction(&mut self, search_direction: P) {
        self.search_direction = Some(search_direction);
    }

    /// Set initial alpha value
    fn set_init_alpha(&mut self, alpha: F) -> Result<(), Error> {
        self.c_x_init = alpha;
        Ok(())
    }
}

impl<P, O, F> Solver<O> for HagerZhangLineSearch<P, F>
where
    O: ArgminOp<Param = P, Output = F, Float = F>,
    P: Clone + SerializeAlias + ArgminDot<P, F> + ArgminScaledAdd<P, F, P>,
    F: ArgminFloat,
{
    const NAME: &'static str = "Hager-Zhang Line search";

    fn init(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &mut IterState<O>,
    ) -> Result<Option<ArgminIterData<O>>, Error> {
        if self.sigma < self.delta {
            return Err(ArgminError::InvalidParameter {
                text: "HagerZhangLineSearch: sigma must be >= delta.".to_string(),
            }
            .into());
        }

        check_param!(
            self.search_direction,
            "HagerZhangLineSearch: Search direction not initialized. Call `set_search_direction`."
        );

        self.init_param = state.get_param();

        let cost = state.get_cost();
        self.finit = if cost.is_infinite() {
            op.apply(self.init_param.as_ref().unwrap())?
        } else {
            cost
        };

        self.init_grad = Some(
            state
                .take_grad()
                .map(Result::Ok)
                .unwrap_or_else(|| op.gradient(self.init_param.as_ref().unwrap()))?,
        );

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

        self.dginit = self
            .init_grad
            .as_ref()
            .unwrap()
            .dot(self.search_direction.as_ref().unwrap());

        self.set_best();
        let new_param = self
            .init_param
            .as_ref()
            .unwrap()
            .scaled_add(&self.best_x, self.search_direction.as_ref().unwrap());
        let best_f = self.best_f;

        Ok(Some(ArgminIterData::new().param(new_param).cost(best_f)))
    }

    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        _state: &mut IterState<O>,
    ) -> Result<ArgminIterData<O>, Error> {
        // L1
        let aa = (self.a_x, self.a_f, self.a_g);
        let bb = (self.b_x, self.b_f, self.b_g);
        let ((mut at_x, mut at_f, mut at_g), (mut bt_x, mut bt_f, mut bt_g)) =
            self.secant2(op, aa, bb)?;

        // L2
        if bt_x - at_x > self.gamma * (self.b_x - self.a_x) {
            let c_x = (at_x + bt_x) / F::from_f64(2.0).unwrap();
            let tmp = self
                .init_param
                .as_ref()
                .unwrap()
                .scaled_add(&c_x, self.search_direction.as_ref().unwrap());
            let c_f = op.apply(&tmp)?;
            let grad = op.gradient(&tmp)?;
            let c_g = self.search_direction.as_ref().unwrap().dot(&grad);
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
            .as_ref()
            .unwrap()
            .scaled_add(&self.best_x, self.search_direction.as_ref().unwrap());
        Ok(ArgminIterData::new().param(new_param).cost(self.best_f))
    }

    fn terminate(&mut self, _state: &IterState<O>) -> TerminationReason {
        if self.best_f - self.finit <= self.delta * self.best_x * self.dginit
            && self.best_g >= self.sigma * self.dginit
        {
            return TerminationReason::LineSearchConditionMet;
        }
        if (F::from_f64(2.0).unwrap() * self.delta - F::from_f64(1.0).unwrap()) * self.dginit
            >= self.best_g
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
    use crate::core::MinimalNoOperator;
    use crate::test_trait_impl;

    test_trait_impl!(hagerzhang, HagerZhangLineSearch<MinimalNoOperator, f64>);
}
