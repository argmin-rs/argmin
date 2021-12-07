// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # References:
//!
//! [0] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
//! Springer. ISBN 0-387-30303-0.

use crate::core::{ArgminDot, ArgminError, ArgminFloat, Error};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

/// Needs to be implemented by everything that wants to be a LineSearchCondition
pub trait LineSearchCondition<T, F>: Serialize {
    /// Evaluate the condition
    fn eval(
        &self,
        cur_cost: F,
        cur_grad: &T,
        init_cost: F,
        init_grad: &T,
        search_direction: &T,
        alpha: F,
    ) -> bool;

    /// Indicates whether this condition requires the computation of the gradient at the new point
    fn requires_cur_grad(&self) -> bool;
}

/// Armijo Condition
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct ArmijoCondition<F> {
    c: F,
}

impl<F: ArgminFloat> ArmijoCondition<F> {
    /// Constructor
    pub fn new(c: F) -> Result<Self, Error> {
        if c <= F::from_f64(0.0).unwrap() || c >= F::from_f64(1.0).unwrap() {
            return Err(ArgminError::InvalidParameter {
                text: "ArmijoCondition: Parameter c must be in (0, 1)".to_string(),
            }
            .into());
        }
        Ok(ArmijoCondition { c })
    }
}

impl<T, F> LineSearchCondition<T, F> for ArmijoCondition<F>
where
    T: ArgminDot<T, F>,
    F: ArgminFloat + Serialize + DeserializeOwned,
{
    fn eval(
        &self,
        cur_cost: F,
        _cur_grad: &T,
        init_cost: F,
        init_grad: &T,
        search_direction: &T,
        alpha: F,
    ) -> bool {
        cur_cost <= init_cost + self.c * alpha * init_grad.dot(search_direction)
    }

    fn requires_cur_grad(&self) -> bool {
        false
    }
}

/// Wolfe Condition
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct WolfeCondition<F> {
    c1: F,
    c2: F,
}

impl<F: ArgminFloat> WolfeCondition<F> {
    /// Constructor
    pub fn new(c1: F, c2: F) -> Result<Self, Error> {
        if c1 <= F::from_f64(0.0).unwrap() || c1 >= F::from_f64(1.0).unwrap() {
            return Err(ArgminError::InvalidParameter {
                text: "WolfeCondition: Parameter c1 must be in (0, 1)".to_string(),
            }
            .into());
        }
        if c2 <= c1 || c2 >= F::from_f64(1.0).unwrap() {
            return Err(ArgminError::InvalidParameter {
                text: "WolfeCondition: Parameter c2 must be in (c1, 1)".to_string(),
            }
            .into());
        }
        Ok(WolfeCondition { c1, c2 })
    }
}

impl<T, F> LineSearchCondition<T, F> for WolfeCondition<F>
where
    T: Clone + ArgminDot<T, F>,
    F: ArgminFloat + DeserializeOwned + Serialize,
{
    fn eval(
        &self,
        cur_cost: F,
        cur_grad: &T,
        init_cost: F,
        init_grad: &T,
        search_direction: &T,
        alpha: F,
    ) -> bool {
        let tmp = init_grad.dot(search_direction);
        (cur_cost <= init_cost + self.c1 * alpha * tmp)
            && cur_grad.dot(search_direction) >= self.c2 * tmp
    }

    fn requires_cur_grad(&self) -> bool {
        true
    }
}

/// Strong Wolfe conditions
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct StrongWolfeCondition<F> {
    c1: F,
    c2: F,
}

impl<F: ArgminFloat> StrongWolfeCondition<F> {
    /// Constructor
    pub fn new(c1: F, c2: F) -> Result<Self, Error> {
        if c1 <= F::from_f64(0.0).unwrap() || c1 >= F::from_f64(1.0).unwrap() {
            return Err(ArgminError::InvalidParameter {
                text: "StrongWolfeCondition: Parameter c1 must be in (0, 1)".to_string(),
            }
            .into());
        }
        if c2 <= c1 || c2 >= F::from_f64(1.0).unwrap() {
            return Err(ArgminError::InvalidParameter {
                text: "StrongWolfeCondition: Parameter c2 must be in (c1, 1)".to_string(),
            }
            .into());
        }
        Ok(StrongWolfeCondition { c1, c2 })
    }
}

impl<T, F> LineSearchCondition<T, F> for StrongWolfeCondition<F>
where
    T: Clone + ArgminDot<T, F>,
    F: ArgminFloat + Serialize + DeserializeOwned,
{
    fn eval(
        &self,
        cur_cost: F,
        cur_grad: &T,
        init_cost: F,
        init_grad: &T,
        search_direction: &T,
        alpha: F,
    ) -> bool {
        let tmp = init_grad.dot(search_direction);
        (cur_cost <= init_cost + self.c1 * alpha * tmp)
            && cur_grad.dot(search_direction).abs() <= self.c2 * tmp.abs()
    }

    fn requires_cur_grad(&self) -> bool {
        true
    }
}

/// Goldstein conditions
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct GoldsteinCondition<F> {
    c: F,
}

impl<F: ArgminFloat> GoldsteinCondition<F> {
    /// Constructor
    pub fn new(c: F) -> Result<Self, Error> {
        if c <= F::from_f64(0.0).unwrap() || c >= F::from_f64(0.5).unwrap() {
            return Err(ArgminError::InvalidParameter {
                text: "GoldsteinCondition: Parameter c must be in (0, 0.5)".to_string(),
            }
            .into());
        }
        Ok(GoldsteinCondition { c })
    }
}

impl<T, F> LineSearchCondition<T, F> for GoldsteinCondition<F>
where
    T: ArgminDot<T, F>,
    F: ArgminFloat + Serialize + DeserializeOwned,
{
    fn eval(
        &self,
        cur_cost: F,
        _cur_grad: &T,
        init_cost: F,
        init_grad: &T,
        search_direction: &T,
        alpha: F,
    ) -> bool {
        let tmp = alpha * init_grad.dot(search_direction);
        init_cost + (F::from_f64(1.0).unwrap() - self.c) * tmp <= cur_cost
            && cur_cost <= init_cost + self.c * alpha * tmp
    }

    fn requires_cur_grad(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_trait_impl;

    test_trait_impl!(goldstein, GoldsteinCondition<f64>);
    test_trait_impl!(armijo, ArmijoCondition<f64>);
    test_trait_impl!(wolfe, WolfeCondition<f64>);
    test_trait_impl!(strongwolfe, StrongWolfeCondition<f64>);
}
