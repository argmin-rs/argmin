// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use {ArgminDot, ArgminError, Error};

/// Needs to be implemented by everything that wants to be a LineSearchCondition
pub trait LineSearchCondition<T> {
    /// Evaluate the condition
    fn eval(&self, f64, T, f64, T, T, f64) -> bool;

    /// Indicates whether this condition requires the computation of the gradient at the new point
    fn requires_cur_grad(&self) -> bool;
}

/// Armijo Condition
pub struct ArmijoCondition {
    c: f64,
}

impl ArmijoCondition {
    /// Constructor
    pub fn new(c: f64) -> Result<Self, Error> {
        if c <= 0.0 || c >= 1.0 {
            return Err(ArgminError::InvalidParameter {
                parameter: "ArmijoCondition: Parameter c must be in (0, 1)".to_string(),
            }.into());
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
        cur_cost <= init_cost + self.c * alpha * init_grad.dot(search_direction)
    }

    fn requires_cur_grad(&self) -> bool {
        false
    }
}

/// Wolfe Condition
pub struct WolfeCondition {
    c1: f64,
    c2: f64,
}

impl WolfeCondition {
    /// Constructor
    pub fn new(c1: f64, c2: f64) -> Result<Self, Error> {
        if c1 <= 0.0 || c1 >= 1.0 {
            return Err(ArgminError::InvalidParameter {
                parameter: "WolfeCondition: Parameter c1 must be in (0, 1)".to_string(),
            }.into());
        }
        if c2 <= c1 || c2 >= 1.0 {
            return Err(ArgminError::InvalidParameter {
                parameter: "WolfeCondition: Parameter c2 must be in (c1, 1)".to_string(),
            }.into());
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
        let tmp = init_grad.dot(search_direction.clone());
        (cur_cost <= init_cost + self.c1 * alpha * tmp)
            && cur_grad.dot(search_direction) >= self.c2 * tmp
    }

    fn requires_cur_grad(&self) -> bool {
        true
    }
}

/// Strong Wolfe conditions
pub struct StrongWolfeCondition {
    c1: f64,
    c2: f64,
}

impl StrongWolfeCondition {
    /// Constructor
    pub fn new(c1: f64, c2: f64) -> Result<Self, Error> {
        if c1 <= 0.0 || c1 >= 1.0 {
            return Err(ArgminError::InvalidParameter {
                parameter: "StrongWolfeCondition: Parameter c1 must be in (0, 1)".to_string(),
            }.into());
        }
        if c2 <= c1 || c2 >= 1.0 {
            return Err(ArgminError::InvalidParameter {
                parameter: "StrongWolfeCondition: Parameter c2 must be in (c1, 1)".to_string(),
            }.into());
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
        let tmp = init_grad.dot(search_direction.clone());
        (cur_cost <= init_cost + self.c1 * alpha * tmp)
            && cur_grad.dot(search_direction).abs() <= self.c2 * tmp.abs()
    }

    fn requires_cur_grad(&self) -> bool {
        true
    }
}

/// Goldstein conditions
pub struct GoldsteinCondition {
    c: f64,
}

impl GoldsteinCondition {
    /// Constructor
    pub fn new(c: f64) -> Result<Self, Error> {
        if c <= 0.0 || c >= 0.5 {
            return Err(ArgminError::InvalidParameter {
                parameter: "GoldsteinCondition: Parameter c must be in (0, 0.5)".to_string(),
            }.into());
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
        let tmp = alpha * init_grad.dot(search_direction);
        init_cost + (1.0 - self.c) * tmp <= cur_cost && cur_cost <= init_cost + self.c * alpha * tmp
    }

    fn requires_cur_grad(&self) -> bool {
        false
    }
}
