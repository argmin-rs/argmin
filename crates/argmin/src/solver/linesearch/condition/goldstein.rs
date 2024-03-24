// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use super::LineSearchCondition;
use crate::core::{ArgminFloat, Error};
use argmin_math::ArgminDot;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// # Goldstein conditions
///
/// Ensures sufficient decrease of the cost function value and also bounds the step length from
/// below.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct GoldsteinCondition<F> {
    c: F,
}

impl<F> GoldsteinCondition<F>
where
    F: ArgminFloat,
{
    /// Construct a new instance of [`GoldsteinCondition`].
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::linesearch::condition::GoldsteinCondition;
    /// let goldstein = GoldsteinCondition::new(0.1f64);
    /// ```
    pub fn new(c: F) -> Result<Self, Error> {
        if c <= float!(0.0) || c >= float!(0.5) {
            return Err(argmin_error!(
                InvalidParameter,
                "GoldsteinCondition: Parameter c must be in (0, 0.5)"
            ));
        }
        Ok(GoldsteinCondition { c })
    }
}

impl<T, G, F> LineSearchCondition<T, G, F> for GoldsteinCondition<F>
where
    G: ArgminDot<T, F>,
    F: ArgminFloat,
{
    fn evaluate_condition(
        &self,
        current_cost: F,
        _current_gradient: Option<&G>,
        initial_cost: F,
        initial_gradient: &G,
        search_direction: &T,
        step_length: F,
    ) -> bool {
        let tmp = step_length * initial_gradient.dot(search_direction);
        initial_cost + (float!(1.0) - self.c) * tmp <= current_cost
            && current_cost <= initial_cost + self.c * tmp
    }

    fn requires_current_gradient(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::ArgminError;

    test_trait_impl!(goldstein, GoldsteinCondition<f64>);

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
                cond.evaluate_condition(
                    f(initial_x + alpha, initial_y),
                    Some(&g(initial_x + alpha, initial_y)),
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
