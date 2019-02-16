// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # References:
//!
//! [0] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
//! Springer. ISBN 0-387-30303-0.

use crate::{ArgminDot, ArgminError, Error};
use serde::{Deserialize, Serialize};

/// Needs to be implemented by everything that wants to be a LineSearchCondition
pub trait LineSearchCondition<T>: Serialize {
    /// Evaluate the condition
    fn eval(
        &self,
        cur_cost: f64,
        cur_grad: T,
        init_cost: f64,
        init_grad: T,
        search_direction: T,
        alpha: f64,
    ) -> bool;

    /// Indicates whether this condition requires the computation of the gradient at the new point
    fn requires_cur_grad(&self) -> bool;
}

/// Armijo Condition
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct ArmijoCondition {
    c: f64,
}

impl ArmijoCondition {
    /// Constructor
    pub fn new(c: f64) -> Result<Self, Error> {
        if c <= 0.0 || c >= 1.0 {
            return Err(ArgminError::InvalidParameter {
                text: "ArmijoCondition: Parameter c must be in (0, 1)".to_string(),
            }
            .into());
        }
        Ok(ArmijoCondition { c })
    }
}

impl<T> LineSearchCondition<T> for ArmijoCondition
where
    T: ArgminDot<T, f64>,
{
    fn eval(
        &self,
        cur_cost: f64,
        _cur_grad: T,
        init_cost: f64,
        init_grad: T,
        search_direction: T,
        alpha: f64,
    ) -> bool {
        cur_cost <= init_cost + self.c * alpha * init_grad.dot(&search_direction)
    }

    fn requires_cur_grad(&self) -> bool {
        false
    }
}

/// Wolfe Condition
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct WolfeCondition {
    c1: f64,
    c2: f64,
}

impl WolfeCondition {
    /// Constructor
    pub fn new(c1: f64, c2: f64) -> Result<Self, Error> {
        if c1 <= 0.0 || c1 >= 1.0 {
            return Err(ArgminError::InvalidParameter {
                text: "WolfeCondition: Parameter c1 must be in (0, 1)".to_string(),
            }
            .into());
        }
        if c2 <= c1 || c2 >= 1.0 {
            return Err(ArgminError::InvalidParameter {
                text: "WolfeCondition: Parameter c2 must be in (c1, 1)".to_string(),
            }
            .into());
        }
        Ok(WolfeCondition { c1, c2 })
    }
}

impl<T> LineSearchCondition<T> for WolfeCondition
where
    T: Clone + ArgminDot<T, f64>,
{
    fn eval(
        &self,
        cur_cost: f64,
        cur_grad: T,
        init_cost: f64,
        init_grad: T,
        search_direction: T,
        alpha: f64,
    ) -> bool {
        let tmp = init_grad.dot(&search_direction);
        (cur_cost <= init_cost + self.c1 * alpha * tmp)
            && cur_grad.dot(&search_direction) >= self.c2 * tmp
    }

    fn requires_cur_grad(&self) -> bool {
        true
    }
}

/// Strong Wolfe conditions
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct StrongWolfeCondition {
    c1: f64,
    c2: f64,
}

impl StrongWolfeCondition {
    /// Constructor
    pub fn new(c1: f64, c2: f64) -> Result<Self, Error> {
        if c1 <= 0.0 || c1 >= 1.0 {
            return Err(ArgminError::InvalidParameter {
                text: "StrongWolfeCondition: Parameter c1 must be in (0, 1)".to_string(),
            }
            .into());
        }
        if c2 <= c1 || c2 >= 1.0 {
            return Err(ArgminError::InvalidParameter {
                text: "StrongWolfeCondition: Parameter c2 must be in (c1, 1)".to_string(),
            }
            .into());
        }
        Ok(StrongWolfeCondition { c1, c2 })
    }
}

impl<T> LineSearchCondition<T> for StrongWolfeCondition
where
    T: Clone + ArgminDot<T, f64>,
{
    fn eval(
        &self,
        cur_cost: f64,
        cur_grad: T,
        init_cost: f64,
        init_grad: T,
        search_direction: T,
        alpha: f64,
    ) -> bool {
        let tmp = init_grad.dot(&search_direction);
        (cur_cost <= init_cost + self.c1 * alpha * tmp)
            && cur_grad.dot(&search_direction).abs() <= self.c2 * tmp.abs()
    }

    fn requires_cur_grad(&self) -> bool {
        true
    }
}

/// Goldstein conditions
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct GoldsteinCondition {
    c: f64,
}

impl GoldsteinCondition {
    /// Constructor
    pub fn new(c: f64) -> Result<Self, Error> {
        if c <= 0.0 || c >= 0.5 {
            return Err(ArgminError::InvalidParameter {
                text: "GoldsteinCondition: Parameter c must be in (0, 0.5)".to_string(),
            }
            .into());
        }
        Ok(GoldsteinCondition { c })
    }
}

impl<T> LineSearchCondition<T> for GoldsteinCondition
where
    T: ArgminDot<T, f64>,
{
    fn eval(
        &self,
        cur_cost: f64,
        _cur_grad: T,
        init_cost: f64,
        init_grad: T,
        search_direction: T,
        alpha: f64,
    ) -> bool {
        let tmp = alpha * init_grad.dot(&search_direction);
        init_cost + (1.0 - self.c) * tmp <= cur_cost && cur_cost <= init_cost + self.c * alpha * tmp
    }

    fn requires_cur_grad(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::send_sync_test;

    send_sync_test!(goldstein, GoldsteinCondition);
    send_sync_test!(armijo, ArmijoCondition);
    send_sync_test!(wolfe, WolfeCondition);
    send_sync_test!(strongwolfe, StrongWolfeCondition);
}
