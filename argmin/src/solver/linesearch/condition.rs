// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # References:
//!
//! \[0\] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
//! Springer. ISBN 0-387-30303-0.

use crate::core::{ArgminError, ArgminFloat, DeserializeOwnedAlias, Error, SerializeAlias};
use crate::prelude::ArgminDot;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// Needs to be implemented by everything that wants to be a LineSearchCondition
pub trait LineSearchCondition<T, F>: SerializeAlias {
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
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
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
    F: ArgminFloat + SerializeAlias + DeserializeOwnedAlias,
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
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
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
    F: ArgminFloat + DeserializeOwnedAlias + SerializeAlias,
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
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
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
    F: ArgminFloat + SerializeAlias + DeserializeOwnedAlias,
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
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
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
    F: ArgminFloat + SerializeAlias + DeserializeOwnedAlias,
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
            && cur_cost <= init_cost + self.c * tmp
    }

    fn requires_cur_grad(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_error;
    use crate::test_trait_impl;

    test_trait_impl!(goldstein, GoldsteinCondition<f64>);
    test_trait_impl!(armijo, ArmijoCondition<f64>);
    test_trait_impl!(wolfe, WolfeCondition<f64>);
    test_trait_impl!(strongwolfe, StrongWolfeCondition<f64>);

    #[test]
    fn test_armijo_new() {
        let c: f64 = 0.01;
        let ArmijoCondition { c: c_arm } = ArmijoCondition::new(c).unwrap();
        assert_eq!(c.to_ne_bytes(), c_arm.to_ne_bytes());

        assert_error!(
            ArmijoCondition::new(1.0f64),
            ArgminError,
            "Invalid parameter: \"ArmijoCondition: Parameter c must be in (0, 1)\""
        );

        assert_error!(
            ArmijoCondition::new(2.0f64),
            ArgminError,
            "Invalid parameter: \"ArmijoCondition: Parameter c must be in (0, 1)\""
        );

        assert_error!(
            ArmijoCondition::new(0.0f64),
            ArgminError,
            "Invalid parameter: \"ArmijoCondition: Parameter c must be in (0, 1)\""
        );

        assert_error!(
            ArmijoCondition::new(-1.0f64),
            ArgminError,
            "Invalid parameter: \"ArmijoCondition: Parameter c must be in (0, 1)\""
        );
    }

    #[test]
    fn test_armijo() {
        let c: f64 = 0.50;
        let cond = ArmijoCondition::new(c).unwrap();
        let f = |x: f64, y: f64| x.powf(2.0) + y.powf(2.0);
        let g = |x: f64, y: f64| vec![2.0 * x, 2.0 * y];
        let initial_x = -1.0;
        let initial_y = -0.0;
        let search_direction = vec![1.0, 0.0];
        for (alpha, acc) in [
            (0.001, true),
            (0.03, true),
            (0.2, true),
            (0.5, true),
            (0.9, true),
            (0.99, true),
            (1.0, true),
            (1.0 + f64::EPSILON, false),
            (1.5, false),
            (1.8, false),
            (2.0, false),
            (2.3, false),
        ] {
            assert_eq!(
                cond.eval(
                    f(initial_x + alpha, initial_y),
                    &g(initial_x + alpha, initial_y),
                    f(initial_x, initial_y),
                    &g(initial_x, initial_y),
                    &search_direction,
                    alpha,
                ),
                acc
            );
        }
    }

    #[test]
    fn test_wolfe_new() {
        let c1: f64 = 0.01;
        let c2: f64 = 0.08;
        let WolfeCondition {
            c1: c1_wolfe,
            c2: c2_wolfe,
        } = WolfeCondition::new(c1, c2).unwrap();
        assert_eq!(c1.to_ne_bytes(), c1_wolfe.to_ne_bytes());
        assert_eq!(c2.to_ne_bytes(), c2_wolfe.to_ne_bytes());

        // c1
        assert_error!(
            WolfeCondition::new(1.0, 0.5),
            ArgminError,
            "Invalid parameter: \"WolfeCondition: Parameter c1 must be in (0, 1)\""
        );

        assert_error!(
            WolfeCondition::new(0.0, 0.5),
            ArgminError,
            "Invalid parameter: \"WolfeCondition: Parameter c1 must be in (0, 1)\""
        );

        assert_error!(
            WolfeCondition::new(-1.0, 0.5),
            ArgminError,
            "Invalid parameter: \"WolfeCondition: Parameter c1 must be in (0, 1)\""
        );

        assert_error!(
            WolfeCondition::new(2.0, 0.5),
            ArgminError,
            "Invalid parameter: \"WolfeCondition: Parameter c1 must be in (0, 1)\""
        );

        // c2
        assert_error!(
            WolfeCondition::new(0.5, -1.0),
            ArgminError,
            "Invalid parameter: \"WolfeCondition: Parameter c2 must be in (c1, 1)\""
        );

        assert_error!(
            WolfeCondition::new(0.5, -0.0),
            ArgminError,
            "Invalid parameter: \"WolfeCondition: Parameter c2 must be in (c1, 1)\""
        );

        assert_error!(
            WolfeCondition::new(0.5, 0.5),
            ArgminError,
            "Invalid parameter: \"WolfeCondition: Parameter c2 must be in (c1, 1)\""
        );

        assert_error!(
            WolfeCondition::new(0.5, 1.0),
            ArgminError,
            "Invalid parameter: \"WolfeCondition: Parameter c2 must be in (c1, 1)\""
        );

        assert_error!(
            WolfeCondition::new(0.5, 2.0),
            ArgminError,
            "Invalid parameter: \"WolfeCondition: Parameter c2 must be in (c1, 1)\""
        );
    }

    #[test]
    fn test_wolfe() {
        let c1: f64 = 0.5;
        let c2: f64 = 0.9;
        let cond = WolfeCondition::new(c1, c2).unwrap();
        let f = |x: f64, y: f64| x.powf(2.0) + y.powf(2.0);
        let g = |x: f64, y: f64| vec![2.0 * x, 2.0 * y];
        let initial_x = -1.0;
        let initial_y = -0.0;
        let search_direction = vec![1.0, 0.0];
        for (alpha, acc) in [
            (0.001, false),
            (0.03, false),
            (0.1 - f64::EPSILON, false),
            (0.1, true),
            (0.5, true),
            (0.9, true),
            (0.99, true),
            (1.0, true),
            (1.0 + f64::EPSILON, false),
            (1.5, false),
            (1.8, false),
            (2.0, false),
            (2.3, false),
        ] {
            assert_eq!(
                cond.eval(
                    f(initial_x + alpha, initial_y),
                    &g(initial_x + alpha, initial_y),
                    f(initial_x, initial_y),
                    &g(initial_x, initial_y),
                    &search_direction,
                    alpha,
                ),
                acc
            );
        }
    }

    #[test]
    fn test_strongwolfe_new() {
        let c1: f64 = 0.01;
        let c2: f64 = 0.08;
        let StrongWolfeCondition {
            c1: c1_wolfe,
            c2: c2_wolfe,
        } = StrongWolfeCondition::new(c1, c2).unwrap();
        assert_eq!(c1.to_ne_bytes(), c1_wolfe.to_ne_bytes());
        assert_eq!(c2.to_ne_bytes(), c2_wolfe.to_ne_bytes());

        // c1
        assert_error!(
            StrongWolfeCondition::new(1.0, 0.5),
            ArgminError,
            "Invalid parameter: \"StrongWolfeCondition: Parameter c1 must be in (0, 1)\""
        );

        assert_error!(
            StrongWolfeCondition::new(0.0, 0.5),
            ArgminError,
            "Invalid parameter: \"StrongWolfeCondition: Parameter c1 must be in (0, 1)\""
        );

        assert_error!(
            StrongWolfeCondition::new(-1.0, 0.5),
            ArgminError,
            "Invalid parameter: \"StrongWolfeCondition: Parameter c1 must be in (0, 1)\""
        );

        assert_error!(
            StrongWolfeCondition::new(2.0, 0.5),
            ArgminError,
            "Invalid parameter: \"StrongWolfeCondition: Parameter c1 must be in (0, 1)\""
        );

        // c2
        assert_error!(
            StrongWolfeCondition::new(0.5, -1.0),
            ArgminError,
            "Invalid parameter: \"StrongWolfeCondition: Parameter c2 must be in (c1, 1)\""
        );

        assert_error!(
            StrongWolfeCondition::new(0.5, 0.0),
            ArgminError,
            "Invalid parameter: \"StrongWolfeCondition: Parameter c2 must be in (c1, 1)\""
        );

        assert_error!(
            StrongWolfeCondition::new(0.5, 0.5),
            ArgminError,
            "Invalid parameter: \"StrongWolfeCondition: Parameter c2 must be in (c1, 1)\""
        );

        assert_error!(
            StrongWolfeCondition::new(0.5, 1.0),
            ArgminError,
            "Invalid parameter: \"StrongWolfeCondition: Parameter c2 must be in (c1, 1)\""
        );

        assert_error!(
            StrongWolfeCondition::new(0.5, 2.0),
            ArgminError,
            "Invalid parameter: \"StrongWolfeCondition: Parameter c2 must be in (c1, 1)\""
        );
    }

    #[test]
    fn test_strongwolfe() {
        // Armijo basically never active (c1 so low that only constraint on gradients have impact
        // on the chosen function).
        let c1: f64 = 0.01;
        let c2: f64 = 0.9;
        let cond = StrongWolfeCondition::new(c1, c2).unwrap();
        let f = |x: f64, y: f64| x.powf(2.0) + y.powf(2.0);
        let g = |x: f64, y: f64| vec![2.0 * x, 2.0 * y];
        let initial_x = -1.0;
        let initial_y = -0.0;
        let search_direction = vec![1.0, 0.0];
        for (alpha, acc) in [
            (0.001, false),
            (0.03, false),
            (0.1 - f64::EPSILON, false),
            (0.1, true),
            (0.15, true),
            (0.9, true),
            (0.99, true),
            (1.0, true),
            (1.9, true),
            (1.9 + f64::EPSILON, false),
            (2.0, false),
            (2.3, false),
        ] {
            assert_eq!(
                cond.eval(
                    f(initial_x + alpha, initial_y),
                    &g(initial_x + alpha, initial_y),
                    f(initial_x, initial_y),
                    &g(initial_x, initial_y),
                    &search_direction,
                    alpha,
                ),
                acc
            );
        }

        // Armijo active
        let c1: f64 = 0.5;
        let c2: f64 = 0.9;
        let cond = StrongWolfeCondition::new(c1, c2).unwrap();
        let f = |x: f64, y: f64| x.powf(2.0) + y.powf(2.0);
        let g = |x: f64, y: f64| vec![2.0 * x, 2.0 * y];
        let initial_x = -1.0;
        let initial_y = -0.0;
        let search_direction = vec![1.0, 0.0];
        for (alpha, acc) in [
            (0.001, false),
            (0.03, false),
            (0.1 - f64::EPSILON, false),
            (0.1, true),
            (0.15, true),
            (0.9, true),
            (0.99, true),
            (1.0, true),
            (1.0 + f64::EPSILON, false),
            (1.9, false),
            (2.0, false),
            (2.3, false),
        ] {
            assert_eq!(
                cond.eval(
                    f(initial_x + alpha, initial_y),
                    &g(initial_x + alpha, initial_y),
                    f(initial_x, initial_y),
                    &g(initial_x, initial_y),
                    &search_direction,
                    alpha,
                ),
                acc
            );
        }
    }

    #[test]
    fn test_goldstein_new() {
        let c: f64 = 0.01;
        let GoldsteinCondition { c: c_arm } = GoldsteinCondition::new(c).unwrap();
        assert_eq!(c.to_ne_bytes(), c_arm.to_ne_bytes());

        assert_error!(
            GoldsteinCondition::new(0.5f64),
            ArgminError,
            "Invalid parameter: \"GoldsteinCondition: Parameter c must be in (0, 0.5)\""
        );

        assert_error!(
            GoldsteinCondition::new(1.0f64),
            ArgminError,
            "Invalid parameter: \"GoldsteinCondition: Parameter c must be in (0, 0.5)\""
        );

        assert_error!(
            GoldsteinCondition::new(0.0f64),
            ArgminError,
            "Invalid parameter: \"GoldsteinCondition: Parameter c must be in (0, 0.5)\""
        );

        assert_error!(
            GoldsteinCondition::new(-1.0f64),
            ArgminError,
            "Invalid parameter: \"GoldsteinCondition: Parameter c must be in (0, 0.5)\""
        );
    }

    #[test]
    fn test_goldstein() {
        let c: f64 = 0.1;
        let cond = GoldsteinCondition::new(c).unwrap();
        let f = |x: f64, y: f64| x.powf(2.0) + y.powf(2.0);
        let g = |x: f64, y: f64| vec![2.0 * x, 2.0 * y];
        let initial_x = -1.0;
        let initial_y = -0.0;
        let search_direction = vec![1.0, 0.0];
        // Need a larger epsilon here, I suppose because of more severe round off errors.
        for (alpha, acc) in [
            (0.001, false),
            (0.03, false),
            (0.2 - 6.0 * f64::EPSILON, false),
            (0.2, true),
            (0.2, true),
            (0.5, true),
            (0.9, true),
            (0.99, true),
            (1.0, true),
            (1.5, true),
            (1.8 - f64::EPSILON, true),
            (1.8, false),
            (2.0, false),
            (2.3, false),
        ] {
            assert_eq!(
                cond.eval(
                    f(initial_x + alpha, initial_y),
                    &g(initial_x + alpha, initial_y),
                    f(initial_x, initial_y),
                    &g(initial_x, initial_y),
                    &search_direction,
                    alpha,
                ),
                acc
            );
        }
    }
}
