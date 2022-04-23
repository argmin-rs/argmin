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

/// # Wolfe conditions
///
/// Assures that a step length satisfies a "sufficient decrease" in cost function value (see
/// [`ArmijoCondition`](`crate::solver::linesearch::condition::ArmijoCondition`) as well as that
/// the slope has been reduced sufficiently (thus making it more likely to be close to a local
/// minimum).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct WolfeCondition<F> {
    c1: F,
    c2: F,
}

impl<F> WolfeCondition<F>
where
    F: ArgminFloat,
{
    /// Construct a new instance of [`WolfeCondition`].
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::linesearch::condition::WolfeCondition;
    /// let wolfe = WolfeCondition::new(0.0001f64, 0.1f64);
    /// ```
    pub fn new(c1: F, c2: F) -> Result<Self, Error> {
        if c1 <= float!(0.0) || c1 >= float!(1.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "WolfeCondition: Parameter c1 must be in (0, 1)"
            ));
        }
        if c2 <= c1 || c2 >= float!(1.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "WolfeCondition: Parameter c2 must be in (c1, 1)"
            ));
        }
        Ok(WolfeCondition { c1, c2 })
    }
}

impl<T, G, F> LineSearchCondition<T, G, F> for WolfeCondition<F>
where
    G: ArgminDot<T, F>,
    F: ArgminFloat,
{
    fn evaluate_condition(
        &self,
        current_cost: F,
        current_gradient: Option<&G>,
        initial_cost: F,
        initial_gradient: &G,
        search_direction: &T,
        step_length: F,
    ) -> bool {
        let tmp = initial_gradient.dot(search_direction);
        (current_cost <= initial_cost + self.c1 * step_length * tmp)
            && current_gradient
                .expect(concat!(
                    "Current gradient not supplied to",
                    " `evaluate_condition` method of `WolfeCondition`"
                ))
                .dot(search_direction)
                >= self.c2 * tmp
    }

    fn requires_current_gradient(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_error;
    use crate::core::ArgminError;
    use crate::test_trait_impl;

    test_trait_impl!(wolfe, WolfeCondition<f64>);

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
