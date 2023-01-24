// Copyright 2018-2022 argmin developers
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

/// # Armijo Condition
///
/// Ensures that the step length "sufficiently" decreases the cost function value.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct ArmijoCondition<F> {
    c: F,
}

impl<F> ArmijoCondition<F>
where
    F: ArgminFloat,
{
    /// Construct a new [`ArmijoCondition`] instance.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::linesearch::condition::ArmijoCondition;
    /// let armijo = ArmijoCondition::new(0.0001f64);
    /// ```
    pub fn new(c: F) -> Result<Self, Error> {
        if c <= float!(0.0) || c >= float!(1.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "ArmijoCondition: Parameter c must be in (0, 1)"
            ));
        }
        Ok(ArmijoCondition { c })
    }
}

impl<T, G, F> LineSearchCondition<T, G, F> for ArmijoCondition<F>
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
        current_cost <= initial_cost + self.c * step_length * initial_gradient.dot(search_direction)
    }

    fn requires_current_gradient(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;
    use crate::assert_error;
    use crate::core::ArgminError;
    use crate::test_trait_impl;

    test_trait_impl!(armijo, ArmijoCondition<f64>);

    #[test]
    fn test_armijo_new() {
        let c: f64 = 0.01;
        let ArmijoCondition { c: c_arm } = ArmijoCondition::new(c).unwrap();
        assert_relative_eq!(c, c_arm, epsilon = f64::EPSILON);

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
