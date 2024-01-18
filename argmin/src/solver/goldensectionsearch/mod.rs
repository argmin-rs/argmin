// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Golden-section search
//!
//! The golden-section search is a technique for finding an extremum (minimum or maximum) of a
//! function inside a specified interval.
//!
//! See [`GoldenSectionSearch`] for details.
//!
//! ## Reference
//!
//! <https://en.wikipedia.org/wiki/Golden-section_search>

use crate::core::{
    ArgminFloat, CostFunction, Error, IterState, Problem, Solver, TerminationReason,
    TerminationStatus, KV,
};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

// Golden ratio is actually 1.61803398874989484820, but that is too much precision for f64.
const GOLDEN_RATIO: f64 = 1.618_033_988_749_895;
const G1: f64 = -1.0 + GOLDEN_RATIO;
const G2: f64 = 1.0 - G1;

/// # Golden-section search
///
/// The golden-section search is a technique for finding an extremum (minimum or maximum) of a
/// function inside a specified interval.
///
/// The method operates by successively narrowing the range of values on the specified interval,
/// which makes it relatively slow, but very robust. The technique derives its name from the fact
/// that the algorithm maintains the function values for four points whose three interval widths
/// are in the ratio 2-φ:2φ-3:2-φ where φ is the golden ratio. These ratios are maintained for each
/// iteration and are maximally efficient.
///
/// The `min_bound` and `max_bound` arguments define values that bracket the expected minimum.
///
/// Requires an initial guess which is to be provided via [`Executor`](`crate::core::Executor`)s
/// `configure` method.
///
/// ## Requirements on the optimization problem
///
/// The optimization problem is required to implement [`CostFunction`].
///
/// ## Reference
///
/// <https://en.wikipedia.org/wiki/Golden-section_search>
#[derive(Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct GoldenSectionSearch<F> {
    g1: F,
    g2: F,
    min_bound: F,
    max_bound: F,
    tolerance: F,

    x0: F,
    x1: F,
    x2: F,
    x3: F,
    f1: F,
    f2: F,
}

impl<F> GoldenSectionSearch<F>
where
    F: ArgminFloat,
{
    /// Construct a new instance of [`GoldenSectionSearch`].
    ///
    /// The `min_bound` and `max_bound` arguments define values that bracket the expected minimum.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::goldensectionsearch::GoldenSectionSearch;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// let gss = GoldenSectionSearch::new(-2.5f64, 3.0f64)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(min_bound: F, max_bound: F) -> Result<Self, Error> {
        if max_bound <= min_bound {
            return Err(argmin_error!(
                InvalidParameter,
                "`GoldenSectionSearch`: `min_bound` must be smaller than `max_bound`."
            ));
        }
        Ok(GoldenSectionSearch {
            g1: F::from(G1).unwrap(),
            g2: F::from(G2).unwrap(),
            min_bound,
            max_bound,
            tolerance: F::from(0.01).unwrap(),
            x0: min_bound,
            x1: F::zero(),
            x2: F::zero(),
            x3: max_bound,
            f1: F::zero(),
            f2: F::zero(),
        })
    }

    /// Set tolerance.
    ///
    /// Must be larger than `0` and defaults to `0.01`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::goldensectionsearch::GoldenSectionSearch;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// let gss = GoldenSectionSearch::new(-2.5f64, 3.0f64)?.with_tolerance(0.0001)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_tolerance(mut self, tolerance: F) -> Result<Self, Error> {
        if tolerance <= float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`GoldenSectionSearch`: Tolerance must be larger than 0."
            ));
        }
        self.tolerance = tolerance;
        Ok(self)
    }
}

impl<O, F> Solver<O, IterState<F, (), (), (), (), F>> for GoldenSectionSearch<F>
where
    O: CostFunction<Param = F, Output = F>,
    F: ArgminFloat,
{
    const NAME: &'static str = "Golden-section search";

    fn init(
        &mut self,
        problem: &mut Problem<O>,
        mut state: IterState<F, (), (), (), (), F>,
    ) -> Result<(IterState<F, (), (), (), (), F>, Option<KV>), Error> {
        let init_estimate = state.take_param().ok_or_else(argmin_error_closure!(
            NotInitialized,
            concat!(
                "`GoldenSectionSearch` requires an initial estimate. ",
                "Please provide an initial guess via `Executor`s `configure` method."
            )
        ))?;
        if init_estimate < self.min_bound || init_estimate > self.max_bound {
            Err(argmin_error!(
                InvalidParameter,
                "`GoldenSectionSearch`: Initial estimate must be ∈ [min_bound, max_bound]."
            ))
        } else {
            let ie_min = init_estimate - self.min_bound;
            let max_ie = self.max_bound - init_estimate;
            let (x1, x2) = if max_ie.abs() > ie_min.abs() {
                (init_estimate, init_estimate + self.g2 * max_ie)
            } else {
                (init_estimate - self.g2 * ie_min, init_estimate)
            };
            self.x1 = x1;
            self.x2 = x2;
            self.f1 = problem.cost(&self.x1)?;
            self.f2 = problem.cost(&self.x2)?;
            if self.f1 < self.f2 {
                Ok((state.param(self.x1).cost(self.f1), None))
            } else {
                Ok((state.param(self.x2).cost(self.f2), None))
            }
        }
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        state: IterState<F, (), (), (), (), F>,
    ) -> Result<(IterState<F, (), (), (), (), F>, Option<KV>), Error> {
        if self.f2 < self.f1 {
            self.x0 = self.x1;
            self.x1 = self.x2;
            self.x2 = self.g1 * self.x1 + self.g2 * self.x3;
            self.f1 = self.f2;
            self.f2 = problem.cost(&self.x2)?;
        } else {
            self.x3 = self.x2;
            self.x2 = self.x1;
            self.x1 = self.g1 * self.x2 + self.g2 * self.x0;
            self.f2 = self.f1;
            self.f1 = problem.cost(&self.x1)?;
        }
        if self.f1 < self.f2 {
            Ok((state.param(self.x1).cost(self.f1), None))
        } else {
            Ok((state.param(self.x2).cost(self.f2), None))
        }
    }

    fn terminate(&mut self, _state: &IterState<F, (), (), (), (), F>) -> TerminationStatus {
        if self.tolerance * (self.x1.abs() + self.x2.abs()) >= (self.x3 - self.x0).abs() {
            return TerminationStatus::Terminated(TerminationReason::SolverConverged);
        }
        TerminationStatus::NotTerminated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{ArgminError, State};
    use crate::test_trait_impl;
    use approx::assert_relative_eq;

    #[derive(Clone)]
    struct GssTestProblem {}

    impl CostFunction for GssTestProblem {
        type Param = f64;
        type Output = f64;

        fn cost(&self, x: &Self::Param) -> Result<Self::Output, Error> {
            Ok((x + 3.0) * (x - 1.0).powi(2))
        }
    }

    test_trait_impl!(golden_section_search, GoldenSectionSearch<f64>);

    #[test]
    fn test_new() {
        let GoldenSectionSearch {
            g1,
            g2,
            min_bound,
            max_bound,
            tolerance,
            x0,
            x1,
            x2,
            x3,
            f1,
            f2,
        } = GoldenSectionSearch::new(-2.5f64, 3.0f64).unwrap();

        assert_eq!(g1.to_ne_bytes(), G1.to_ne_bytes());
        assert_eq!(g2.to_ne_bytes(), G2.to_ne_bytes());
        assert_eq!(min_bound.to_ne_bytes(), (-2.5f64).to_ne_bytes());
        assert_eq!(max_bound.to_ne_bytes(), 3.0f64.to_ne_bytes());
        assert_eq!(tolerance.to_ne_bytes(), 0.01f64.to_ne_bytes());
        assert_eq!(x0.to_ne_bytes(), min_bound.to_ne_bytes());
        assert_eq!(x1.to_ne_bytes(), 0f64.to_ne_bytes());
        assert_eq!(x2.to_ne_bytes(), 0f64.to_ne_bytes());
        assert_eq!(x3.to_ne_bytes(), max_bound.to_ne_bytes());
        assert_eq!(f1.to_ne_bytes(), 0f64.to_ne_bytes());
        assert_eq!(f2.to_ne_bytes(), 0f64.to_ne_bytes());
    }

    #[test]
    fn test_new_errors() {
        let res = GoldenSectionSearch::new(2.5f64, -3.0f64);

        assert_error!(
            res,
            ArgminError,
            concat!(
                "Invalid parameter: \"`GoldenSectionSearch`: ",
                "`min_bound` must be smaller than `max_bound`.\""
            )
        );

        let res = GoldenSectionSearch::new(2.5f64, 2.5f64);

        assert_error!(
            res,
            ArgminError,
            concat!(
                "Invalid parameter: \"`GoldenSectionSearch`: ",
                "`min_bound` must be smaller than `max_bound`.\""
            )
        );
    }

    #[test]
    fn test_tolerance() {
        let GoldenSectionSearch { tolerance, .. } = GoldenSectionSearch::new(-2.5f64, 3.0f64)
            .unwrap()
            .with_tolerance(0.001)
            .unwrap();

        assert_eq!(tolerance.to_ne_bytes(), 0.001f64.to_ne_bytes());
    }

    #[test]
    fn test_tolerance_errors() {
        let res = GoldenSectionSearch::new(-2.5f64, 3.0f64)
            .unwrap()
            .with_tolerance(0.0);
        assert_error!(
            res,
            ArgminError,
            "Invalid parameter: \"`GoldenSectionSearch`: Tolerance must be larger than 0.\""
        );

        let res = GoldenSectionSearch::new(-2.5f64, 3.0f64)
            .unwrap()
            .with_tolerance(-1.0);
        assert_error!(
            res,
            ArgminError,
            "Invalid parameter: \"`GoldenSectionSearch`: Tolerance must be larger than 0.\""
        );
    }

    #[test]
    fn test_init_param_not_initialized() {
        let mut gss = GoldenSectionSearch::new(-2.5f64, 3.0f64).unwrap();
        let res = gss.init(&mut Problem::new(GssTestProblem {}), IterState::new());
        assert_error!(
            res,
            ArgminError,
            concat!(
                "Not initialized: \"`GoldenSectionSearch` requires an initial estimate. ",
                "Please provide an initial guess via `Executor`s `configure` method.\""
            )
        );
    }

    #[test]
    fn test_init_param_outside_bounds() {
        let mut gss = GoldenSectionSearch::new(-2.5f64, 3.0f64).unwrap();
        let res = gss.init(
            &mut Problem::new(GssTestProblem {}),
            IterState::new().param(5.0f64),
        );
        assert_error!(
            res,
            ArgminError,
            concat!(
                "Invalid parameter: \"`GoldenSectionSearch`: Initial estimate must be ∈ [min_bound, max_bound].\"",
            )
        );
    }

    #[test]
    fn test_init() {
        let mut gss = GoldenSectionSearch::new(-2.5f64, 3.0f64).unwrap();
        let problem = GssTestProblem {};
        let (state, kv) = gss
            .init(
                &mut Problem::new(problem.clone()),
                IterState::new().param(-0.5f64),
            )
            .unwrap();

        assert!(kv.is_none());

        let GoldenSectionSearch {
            g1,
            g2,
            min_bound,
            max_bound,
            tolerance,
            x0,
            x1,
            x2,
            x3,
            f1,
            f2,
        } = gss.clone();

        assert_relative_eq!(x1, -0.5f64, epsilon = f64::EPSILON);
        assert_relative_eq!(x2, -0.5f64 + g2 * 3.5f64, epsilon = f64::EPSILON);
        assert_relative_eq!(f1, problem.cost(&x1).unwrap(), epsilon = f64::EPSILON);
        assert_relative_eq!(f2, problem.cost(&x2).unwrap(), epsilon = f64::EPSILON);
        if f1 < f2 {
            assert_relative_eq!(*state.param.as_ref().unwrap(), x1, epsilon = f64::EPSILON);
            assert_relative_eq!(state.cost, f1, epsilon = f64::EPSILON);
        } else {
            assert_relative_eq!(*state.param.as_ref().unwrap(), x2, epsilon = f64::EPSILON);
            assert_relative_eq!(state.cost, f2, epsilon = f64::EPSILON);
        }

        assert_eq!(g1.to_ne_bytes(), G1.to_ne_bytes());
        assert_eq!(g2.to_ne_bytes(), G2.to_ne_bytes());
        assert_eq!(min_bound.to_ne_bytes(), (-2.5f64).to_ne_bytes());
        assert_eq!(max_bound.to_ne_bytes(), 3.0f64.to_ne_bytes());
        assert_eq!(tolerance.to_ne_bytes(), 0.01f64.to_ne_bytes());
        assert_eq!(x0.to_ne_bytes(), min_bound.to_ne_bytes());
        assert_eq!(x3.to_ne_bytes(), max_bound.to_ne_bytes());
    }

    #[test]
    fn test_next_iter_1() {
        let mut gss = GoldenSectionSearch::new(-2.5f64, 3.0f64).unwrap();
        let mut problem = Problem::new(GssTestProblem {});

        gss.f1 = 10.0f64;
        gss.f2 = 5.0f64;
        gss.x0 = 0.0f64;
        gss.x1 = 1.0f64;
        gss.x2 = 2.0f64;
        gss.x3 = 3.0f64;

        let (state, kv) = gss
            .next_iter(&mut problem, IterState::new().param(-0.5f64))
            .unwrap();

        assert!(kv.is_none());

        let GoldenSectionSearch {
            g1,
            g2,
            min_bound,
            max_bound,
            tolerance,
            x0,
            x1,
            x2,
            x3,
            f1,
            f2,
        } = gss.clone();

        assert_relative_eq!(x0, 1.0f64, epsilon = f64::EPSILON);
        assert_relative_eq!(x1, 2.0f64, epsilon = f64::EPSILON);
        assert_relative_eq!(x2, g1 * 2.0f64 + g2 * x3, epsilon = f64::EPSILON);
        assert_relative_eq!(f1, 5.0f64, epsilon = f64::EPSILON);
        assert_relative_eq!(f2, problem.cost(&x2).unwrap(), epsilon = f64::EPSILON);
        assert_eq!(g1.to_ne_bytes(), G1.to_ne_bytes());
        assert_eq!(g2.to_ne_bytes(), G2.to_ne_bytes());
        assert_eq!(min_bound.to_ne_bytes(), (-2.5f64).to_ne_bytes());
        assert_eq!(max_bound.to_ne_bytes(), 3.0f64.to_ne_bytes());
        assert_eq!(tolerance.to_ne_bytes(), 0.01f64.to_ne_bytes());
        if f1 < f2 {
            assert_relative_eq!(*state.param.as_ref().unwrap(), x1, epsilon = f64::EPSILON);
            assert_relative_eq!(state.cost, f1, epsilon = f64::EPSILON);
        } else {
            assert_relative_eq!(*state.param.as_ref().unwrap(), x2, epsilon = f64::EPSILON);
            assert_relative_eq!(state.cost, f2, epsilon = f64::EPSILON);
        }
    }

    #[test]
    fn test_next_iter_2() {
        let mut gss = GoldenSectionSearch::new(-2.5f64, 3.0f64).unwrap();
        let mut problem = Problem::new(GssTestProblem {});

        gss.f1 = 5.0f64;
        gss.f2 = 10.0f64;
        gss.x0 = 0.0f64;
        gss.x1 = 1.0f64;
        gss.x2 = 2.0f64;
        gss.x3 = 3.0f64;

        let (state, kv) = gss
            .next_iter(&mut problem, IterState::new().param(-0.5f64))
            .unwrap();

        assert!(kv.is_none());

        let GoldenSectionSearch {
            g1,
            g2,
            min_bound,
            max_bound,
            tolerance,
            x0,
            x1,
            x2,
            x3,
            f1,
            f2,
        } = gss.clone();

        assert_relative_eq!(x0, 0.0f64, epsilon = f64::EPSILON);
        assert_relative_eq!(x1, g1 * x2 + g2 * x0, epsilon = f64::EPSILON);
        assert_relative_eq!(x2, 1.0f64, epsilon = f64::EPSILON);
        assert_relative_eq!(x3, 2.0f64, epsilon = f64::EPSILON);
        assert_relative_eq!(f1, problem.cost(&x1).unwrap(), epsilon = f64::EPSILON);
        assert_relative_eq!(f2, 5.0f64, epsilon = f64::EPSILON);
        assert_eq!(g1.to_ne_bytes(), G1.to_ne_bytes());
        assert_eq!(g2.to_ne_bytes(), G2.to_ne_bytes());
        assert_eq!(min_bound.to_ne_bytes(), (-2.5f64).to_ne_bytes());
        assert_eq!(max_bound.to_ne_bytes(), 3.0f64.to_ne_bytes());
        assert_eq!(tolerance.to_ne_bytes(), 0.01f64.to_ne_bytes());
        if f1 < f2 {
            assert_relative_eq!(*state.param.as_ref().unwrap(), x1, epsilon = f64::EPSILON);
            assert_relative_eq!(state.cost, f1, epsilon = f64::EPSILON);
        } else {
            assert_relative_eq!(*state.param.as_ref().unwrap(), x2, epsilon = f64::EPSILON);
            assert_relative_eq!(state.cost, f2, epsilon = f64::EPSILON);
        }
    }
}
