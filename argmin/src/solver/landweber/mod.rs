// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Landweber iteration
//!
//! The Landweber iteration is a solver for ill-posed linear inverse problems.
//! See [`Landweber`] for details.
//!
//! ## References
//!
//! Landweber, L. (1951): An iteration formula for Fredholm integral equations of the first
//! kind. Amer. J. Math. 73, 615–624
//!
//! <https://en.wikipedia.org/wiki/Landweber_iteration>

use crate::core::{ArgminFloat, Error, Gradient, IterState, Problem, Solver, KV};
use argmin_math::ArgminScaledSub;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// # Landweber iteration
///
/// The Landweber iteration is a solver for ill-posed linear inverse problems.
///
/// In iteration `k`, the new parameter vector `x_{k+1}` is calculated from the previous parameter
/// vector `x_k` and the gradient at `x_k` according to the following update rule:
///
/// `x_{k+1} = x_k - omega * \nabla f(x_k)`
///
/// ## Requirements on the optimization problem
///
/// The optimization problem is required to implement [`Gradient`].
///
/// ## References
///
/// Landweber, L. (1951): An iteration formula for Fredholm integral equations of the first
/// kind. Amer. J. Math. 73, 615–624
///
/// <https://en.wikipedia.org/wiki/Landweber_iteration>
#[derive(Clone, Copy)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct Landweber<F> {
    /// omega
    omega: F,
}

impl<F> Landweber<F> {
    /// Construct a new instance of [`Landweber`]
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::landweber::Landweber;
    /// let omega: f64 = 0.5;
    /// let landweber = Landweber::new(omega);
    /// ```
    pub fn new(omega: F) -> Self {
        Landweber { omega }
    }
}

impl<O, F, P, G> Solver<O, IterState<P, G, (), (), (), F>> for Landweber<F>
where
    O: Gradient<Param = P, Gradient = G>,
    P: Clone + ArgminScaledSub<G, F, P>,
    F: ArgminFloat,
{
    const NAME: &'static str = "Landweber";

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        mut state: IterState<P, G, (), (), (), F>,
    ) -> Result<(IterState<P, G, (), (), (), F>, Option<KV>), Error> {
        let param = state.take_param().ok_or_else(argmin_error_closure!(
            NotInitialized,
            concat!(
                "`Landweber` requires an initial parameter vector. ",
                "Please provide an initial guess via `Executor`s `configure` method."
            )
        ))?;
        let grad = problem.gradient(&param)?;
        let new_param = param.scaled_sub(&self.omega, &grad);
        Ok((state.param(new_param), None))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{test_utils::TestProblem, ArgminError, Problem, State};
    use crate::test_trait_impl;
    use approx::assert_relative_eq;

    test_trait_impl!(landweber, Landweber<f64>);

    #[test]
    fn test_new() {
        let omega_in: f64 = 0.5;
        let Landweber { omega } = Landweber::new(omega_in);
        assert_eq!(omega.to_ne_bytes(), omega_in.to_ne_bytes());
    }

    #[test]
    fn test_next_iter_param_not_initialized() {
        let omega: f64 = 0.5;
        let mut landweber = Landweber::new(omega);
        let state = IterState::new();
        let res = landweber.next_iter(&mut Problem::new(TestProblem::new()), state);
        assert_error!(
            res,
            ArgminError,
            concat!(
                "Not initialized: \"`Landweber` requires an initial parameter vector. ",
                "Please provide an initial guess via `Executor`s `configure` method.\""
            )
        );
    }

    #[test]
    fn test_next_iter() {
        let omega: f64 = 0.5;
        let mut landweber = Landweber::new(omega);
        let state = IterState::new().param(vec![2.0, 4.0]);
        let (state, kv) = landweber
            .next_iter(&mut Problem::new(TestProblem::new()), state)
            .unwrap();
        assert!(kv.is_none());
        let new_param = state.get_param().unwrap();
        assert_relative_eq!(new_param[0], 1.0, epsilon = f64::EPSILON);
        assert_relative_eq!(new_param[1], 2.0, epsilon = f64::EPSILON);
    }
}
