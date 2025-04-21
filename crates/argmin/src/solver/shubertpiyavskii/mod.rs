// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Shubert-Piyavskii method
//!
//! The Shubert-Piyavskii method is a technique for finding an extremum (minimum or maximum) of a
//! univariate, potentially multimodal Lipschitz continuous function inside a specified interval.
//!
//! See [`ShubertPiyavskii`] for details.
//!
//! ## Reference
//!
//! <https://web.stanford.edu/group/sisl/k12/optimization/MO-unit2-pdfs/2.8global2sawtooth.pdf>

use crate::core::{
    ArgminFloat, CostFunction, Error, IterState, Problem, Solver, TerminationReason,
    TerminationStatus, KV,
};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

const DEFAULT_TOL: f64 = 0.01;
const BOUNDS_ERR_MSG: &str = "`ShubertPiyavskii`: `min_bound` must be smaller than `max_bound`.";
const LIPSCHITZ_ERR_MSG: &str = "`ShubertPiyavskii`: Lipschitz constant must be positive.";
const TOLERANCE_ERR_MSG: &str = "`ShubertPiyavskii`: Tolerance must be positive.";

#[inline(always)]
fn nonfinite_err_msg<F: ArgminFloat>(x: F) -> String {
    format!(
        "`ShubertPiyavskii`: Objective returned non-finite value at x = {}; \
        cannot be Lipschitz continuous.",
        x
    )
}

#[derive(Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
struct SearchInterval<F: ArgminFloat> {
    x_lower: F,
    x_upper: F,
    f_lower: F,
    f_upper: F,
    lower_bound: F,
}

impl<F: ArgminFloat> SearchInterval<F> {
    fn new(
        x_lower: F,
        x_upper: F,
        f_lower: F,
        f_upper: F,
        lipschitz_const: F,
    ) -> Result<Self, Error> {
        if !f_lower.is_finite() {
            return Err(argmin_error!(InvalidParameter, nonfinite_err_msg(x_lower)));
        }

        if !f_upper.is_finite() {
            return Err(argmin_error!(InvalidParameter, nonfinite_err_msg(x_upper)));
        }

        let lower_bound = Self::lower_bound(x_lower, x_upper, f_lower, f_upper, lipschitz_const);

        Ok(SearchInterval {
            x_lower,
            x_upper,
            f_lower,
            f_upper,
            lower_bound,
        })
    }

    #[inline]
    fn lower_bound(x_lower: F, x_upper: F, f_lower: F, f_upper: F, lipschitz_const: F) -> F {
        let two = F::from(2.0).unwrap();
        (f_lower + f_upper - lipschitz_const * (x_upper - x_lower)) / two
    }
}

impl<F: ArgminFloat> Ord for SearchInterval<F> {
    fn cmp(&self, other: &Self) -> Ordering {
        other.lower_bound.partial_cmp(&self.lower_bound).unwrap()
    }
}

impl<F: ArgminFloat> PartialOrd for SearchInterval<F> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<F: ArgminFloat> PartialEq for SearchInterval<F> {
    fn eq(&self, other: &Self) -> bool {
        self.lower_bound == other.lower_bound
    }
}

impl<F: ArgminFloat> Eq for SearchInterval<F> {}

/// # Shubert-Piyavskii method
///
/// The Shubert-Piyavskii method (also known as the Sawtooth method) optimizes a univariate,
/// Lipschitz continuous function inside a specified interval. (In this case, we implement a solver
/// to find the minimum of the function, but it can be easily adapted to find the maximum instead.)
///
/// Given an accurate Lipschitz constant for the objective function, the method is guaranteed to
/// come within an arbitrary epsilon of the true global minimum. (A Lipschitz constant is a real
/// number `K ≥ 0` such that for all `x` and `y` in the interval, `|f(x) - f(y)| <= K * |x - y|`.
/// For differentiable functions, this is equivalent to an absolute bound on the derivative.)
///
/// Unlike most other deterministic global optimization algorithms (e.g., golden-section search),
/// the Shubert-Piyavskii method is not restricted to unimodal functions and can be applied to
/// any Lipschitz continuous function. It also does not require an initial guess; instead, it
/// deterministically samples points from ever-narrowing subintervals of the search space:
/// 1. Use the Lipschitz constant to bound the function on each subinterval from below.
/// 2. Evaluate at the point where this bound predicts the minimum.
/// 3. Subdivide and repeat until the solution is sufficiently close to the current bound.
///
/// The `min_bound` and `max_bound` arguments define values that bracket the expected minimum.
///
/// The `lipschitz_const` argument is the Lipschitz constant for the objective function. (A smaller
/// value will yield faster convergence to the global minimum, but any estimate greater than or
/// equal to the true Lipschitz constant will work.) If `lipschitz_const` is set to zero, the solver
/// assumes that the function is constant and converges immediately.
///
/// ## Requirements on the optimization problem
///
/// The optimization problem is required to implement [`CostFunction`].
///
/// ## Reference
///
/// <https://web.stanford.edu/group/sisl/k12/optimization/MO-unit2-pdfs/2.8global2sawtooth.pdf>
#[derive(Clone)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct ShubertPiyavskii<F: ArgminFloat> {
    min_bound: F,
    max_bound: F,
    lipschitz_const: F,
    tolerance: F,
    intervals: BinaryHeap<SearchInterval<F>>,
}

impl<F: ArgminFloat> ShubertPiyavskii<F> {
    /// Construct a new instance of [`ShubertPiyavskii`].
    ///
    /// The `min_bound` and `max_bound` arguments define values that bracket the expected minimum.
    ///
    /// The `lipschitz_const` argument is the Lipschitz constant for the objective function.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::shubertpiyavskii::ShubertPiyavskii;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// let sp = ShubertPiyavskii::new(1.0, 4.0, 2.5)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(min_bound: F, max_bound: F, lipschitz_const: F) -> Result<Self, Error> {
        if max_bound <= min_bound {
            return Err(argmin_error!(InvalidParameter, BOUNDS_ERR_MSG));
        }

        if lipschitz_const < float!(0.0) {
            return Err(argmin_error!(InvalidParameter, LIPSCHITZ_ERR_MSG));
        }

        Ok(ShubertPiyavskii {
            min_bound,
            max_bound,
            lipschitz_const,
            tolerance: F::from(DEFAULT_TOL).unwrap(),
            intervals: BinaryHeap::new(),
        })
    }

    /// Set tolerance.
    ///
    /// Must be larger than `0` and defaults to `0.01`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::shubertpiyavskii::ShubertPiyavskii;
    /// # use argmin::core::Error;
    /// # fn main() -> Result<(), Error> {
    /// let sp = ShubertPiyavskii::new(1.0, 4.0, 2.5)?.with_tolerance(0.0001)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_tolerance(mut self, tolerance: F) -> Result<Self, Error> {
        if tolerance <= float!(0.0) {
            return Err(argmin_error!(InvalidParameter, TOLERANCE_ERR_MSG));
        }

        self.tolerance = tolerance;
        Ok(self)
    }

    /// Sample the expected minimum from the most promising candidate subinterval.
    ///
    /// This point is then used to split the current search interval into two child intervals.
    #[inline]
    fn sample_point(&self, x_lower: F, x_upper: F, f_lower: F, f_upper: F) -> F {
        let two = F::from(2.0).unwrap();
        let x_sample = (x_lower + x_upper - (f_upper - f_lower) / self.lipschitz_const) / two;
        x_sample.clamp(x_lower, x_upper)
    }

    fn split_interval<O: CostFunction<Param = F, Output = F>>(
        &self,
        interval: &SearchInterval<F>,
        problem: &mut Problem<O>,
    ) -> Result<(SearchInterval<F>, SearchInterval<F>), Error> {
        let x_lower = interval.x_lower;
        let x_upper = interval.x_upper;
        let f_lower = interval.f_lower;
        let f_upper = interval.f_upper;
        let x_sample = self.sample_point(x_lower, x_upper, f_lower, f_upper);
        let f_sample = problem.cost(&x_sample)?;

        let child1 =
            SearchInterval::new(x_lower, x_sample, f_lower, f_sample, self.lipschitz_const)?;
        let child2 =
            SearchInterval::new(x_sample, x_upper, f_sample, f_upper, self.lipschitz_const)?;
        Ok((child1, child2))
    }
}

impl<O, F> Solver<O, IterState<F, (), (), (), (), F>> for ShubertPiyavskii<F>
where
    O: CostFunction<Param = F, Output = F>,
    F: ArgminFloat,
{
    fn name(&self) -> &str {
        "Shubert-Piyavskii method"
    }

    fn init(
        &mut self,
        problem: &mut Problem<O>,
        state: IterState<F, (), (), (), (), F>,
    ) -> Result<(IterState<F, (), (), (), (), F>, Option<KV>), Error> {
        let f_lower = problem.cost(&self.min_bound)?;
        let f_upper = problem.cost(&self.max_bound)?;

        let (x_best, f_best) = if f_lower <= f_upper {
            (self.min_bound, f_lower)
        } else {
            (self.max_bound, f_upper)
        };

        let initial_interval = SearchInterval::new(
            self.min_bound,
            self.max_bound,
            f_lower,
            f_upper,
            self.lipschitz_const,
        )?;

        self.intervals.push(initial_interval);
        Ok((state.param(x_best).cost(f_best), None))
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        state: IterState<F, (), (), (), (), F>,
    ) -> Result<(IterState<F, (), (), (), (), F>, Option<KV>), Error> {
        let search_interval = self.intervals.pop().unwrap();
        let mut x_best = state.param.unwrap();
        let mut f_best = state.cost;

        if f_best - search_interval.lower_bound >= self.tolerance {
            let (child1, child2) = self.split_interval(&search_interval, problem)?;
            let x_sample = child1.x_upper;
            let f_sample = child1.f_upper;

            if f_sample <= f_best {
                x_best = x_sample;
                f_best = f_sample;
            }

            for child in [child1, child2] {
                if f_best - child.lower_bound >= self.tolerance {
                    self.intervals.push(child);
                }
            }
        }

        Ok((state.param(x_best).cost(f_best), None))
    }

    fn terminate(&mut self, _state: &IterState<F, (), (), (), (), F>) -> TerminationStatus {
        if self.intervals.is_empty() {
            TerminationStatus::Terminated(TerminationReason::SolverConverged)
        } else {
            TerminationStatus::NotTerminated
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{ArgminError, Executor, State};
    use approx::{assert_relative_eq, relative_eq};
    use std::f64::consts::PI;

    const MIN_BOUND: f64 = 2.0;
    const MAX_BOUND: f64 = 12.0;
    const LIPSCHITZ_CONST: f64 = 5.0;
    const TEST_TOL: f64 = 1e-5;
    // The objective function has global minima at phase shift 5π/16 with period 2π
    const GLOBAL_MINIMIZER: f64 = (37.0 * PI) / 16.0;

    #[derive(Clone)]
    struct SpTestProblem {}

    impl CostFunction for SpTestProblem {
        type Param = f64;
        type Output = f64;

        /// A Lipschitz continuous, highly multimodal function with many non-differentiable edges.
        ///
        /// Local minima occur precisely every π/4 units, but global minima repeat only every 2π
        /// units. The function's highly multimodal behavior and non-differentiable edges make it an
        /// excellent stress test for the Shubert-Piyavskii method.
        fn cost(&self, x: &Self::Param) -> Result<Self::Output, Error> {
            let sin5x = (5.0 * x).sin();
            let cos3x = (3.0 * x).cos();
            Ok(sin5x.max(cos3x))
        }
    }

    test_trait_impl!(shubert_piyavskii, ShubertPiyavskii<f64>);

    /// Validate the `ShubertPiyavskii` constructor.
    #[test]
    fn test_new() {
        let ShubertPiyavskii {
            min_bound,
            max_bound,
            lipschitz_const,
            tolerance,
            intervals,
        } = ShubertPiyavskii::new(MIN_BOUND, MAX_BOUND, LIPSCHITZ_CONST).unwrap();

        assert_eq!(min_bound.to_ne_bytes(), MIN_BOUND.to_ne_bytes());
        assert_eq!(max_bound.to_ne_bytes(), MAX_BOUND.to_ne_bytes());
        assert_eq!(lipschitz_const.to_ne_bytes(), LIPSCHITZ_CONST.to_ne_bytes());
        assert_eq!(tolerance.to_ne_bytes(), DEFAULT_TOL.to_ne_bytes());
        assert!(intervals.is_empty());
    }

    /// Validate that the tolerance parameter is set correctly.
    #[test]
    fn test_tolerance() {
        let ShubertPiyavskii { tolerance, .. } =
            ShubertPiyavskii::new(MIN_BOUND, MAX_BOUND, LIPSCHITZ_CONST).unwrap();

        assert_eq!(tolerance.to_ne_bytes(), DEFAULT_TOL.to_ne_bytes());

        let ShubertPiyavskii { tolerance, .. } =
            ShubertPiyavskii::new(MIN_BOUND, MAX_BOUND, LIPSCHITZ_CONST)
                .unwrap()
                .with_tolerance(TEST_TOL)
                .unwrap();

        assert_eq!(tolerance.to_ne_bytes(), TEST_TOL.to_ne_bytes());
    }

    /// Validate proper handling of bounds errors (`min_bound` >= `max_bound`).
    #[test]
    fn test_bounds_errors() {
        // Error when `min_bound` == `max_bound`
        let res = ShubertPiyavskii::new(0.0, 0.0, LIPSCHITZ_CONST);

        assert_error!(
            res,
            ArgminError,
            format!("Invalid parameter: \"{}\"", BOUNDS_ERR_MSG)
        );

        // Error when `min_bound` > `max_bound`
        let res = ShubertPiyavskii::new(1.0, 0.0, LIPSCHITZ_CONST);

        assert_error!(
            res,
            ArgminError,
            format!("Invalid parameter: \"{}\"", BOUNDS_ERR_MSG)
        );
    }

    /// Validate immediate convergence of the solver when the Lipschitz constant is set to zero.
    ///
    /// When `lipschitz_const` is set to zero, the solver assumes that the function is constant;
    /// certainly, this is not the case here, but the burden of choosing an accurate Lipschitz
    /// constant is on the user.
    #[test]
    fn test_zero_lipschitz() {
        let mut sp = ShubertPiyavskii::new(MIN_BOUND, MAX_BOUND, 0.0).unwrap();
        let mut problem = Problem::new(SpTestProblem {});
        let (mut state, _) = sp.init(&mut problem, IterState::new()).unwrap();
        (state, _) = sp.next_iter(&mut problem, state).unwrap();

        assert_eq!(
            <ShubertPiyavskii<f64> as Solver<
                SpTestProblem,
                IterState<f64, (), (), (), (), f64>,
            >>::terminate(
                &mut sp, &state
            ),
            TerminationStatus::Terminated(TerminationReason::SolverConverged)
        );

        let x_sample = sp.sample_point(
            MIN_BOUND,
            MAX_BOUND,
            problem.cost(&MIN_BOUND).unwrap(),
            problem.cost(&MAX_BOUND).unwrap(),
        );

        let x_best = state.param.unwrap();
        let f_best = state.cost;

        assert_relative_eq!(x_best, &x_sample, epsilon = DEFAULT_TOL);

        assert_relative_eq!(
            f_best,
            problem.cost(&x_sample).unwrap(),
            epsilon = DEFAULT_TOL
        );
    }

    /// Validate proper handling of Lipschitz constant errors (negative values).
    #[test]
    fn test_lipschitz_error() {
        let res = ShubertPiyavskii::new(MIN_BOUND, MAX_BOUND, -LIPSCHITZ_CONST);

        assert_error!(
            res,
            ArgminError,
            format!("Invalid parameter: \"{}\"", LIPSCHITZ_ERR_MSG)
        );
    }

    /// Validate proper handling of tolerance errors (both zero and negative values).
    #[test]
    fn test_tolerance_errors() {
        // Error when `tolerance` == 0
        let res = ShubertPiyavskii::new(MIN_BOUND, MAX_BOUND, LIPSCHITZ_CONST)
            .unwrap()
            .with_tolerance(0.0);

        assert_error!(
            res,
            ArgminError,
            format!("Invalid parameter: \"{}\"", TOLERANCE_ERR_MSG)
        );

        // Error when `tolerance` < 0
        let res = ShubertPiyavskii::new(MIN_BOUND, MAX_BOUND, LIPSCHITZ_CONST)
            .unwrap()
            .with_tolerance(-1.0);

        assert_error!(
            res,
            ArgminError,
            format!("Invalid parameter: \"{}\"", TOLERANCE_ERR_MSG)
        );
    }

    /// Confirm proper initialization of the solver.
    #[test]
    fn test_init() {
        let mut sp = ShubertPiyavskii::new(MIN_BOUND, MAX_BOUND, LIPSCHITZ_CONST).unwrap();
        let problem = SpTestProblem {};
        let (state, kv) = sp
            .init(&mut Problem::new(problem.clone()), IterState::new())
            .unwrap();

        // The solver should not return metadata
        assert!(kv.is_none());

        let ShubertPiyavskii {
            min_bound,
            max_bound,
            lipschitz_const: _,
            tolerance: _,
            intervals,
        } = sp;

        // One interval created upon initialization
        assert_eq!(intervals.len(), 1);

        let x_best = state.param.unwrap();
        let f_best = state.cost;

        assert_relative_eq!(
            f_best,
            problem.cost(&x_best).unwrap(),
            epsilon = f64::EPSILON
        );

        let selected_lower = relative_eq!(x_best, min_bound, epsilon = f64::EPSILON)
            && relative_eq!(
                f_best,
                problem.cost(&min_bound).unwrap(),
                epsilon = f64::EPSILON
            );

        let selected_upper = relative_eq!(x_best, max_bound, epsilon = f64::EPSILON)
            && relative_eq!(
                f_best,
                problem.cost(&max_bound).unwrap(),
                epsilon = f64::EPSILON
            );

        // The current guess should be either the lower or the upper bound
        assert!(selected_lower || selected_upper);
    }

    /// Validate the termination logic of the solver.
    #[test]
    fn test_termination() {
        let mut sp = ShubertPiyavskii::new(MIN_BOUND, MAX_BOUND, LIPSCHITZ_CONST).unwrap();
        let mut problem = Problem::new(SpTestProblem {});
        let (state, _) = sp.init(&mut problem, IterState::new()).unwrap();

        // The solver should not terminate immediately
        assert_eq!(
            <ShubertPiyavskii<f64> as Solver<
                SpTestProblem,
                IterState<f64, (), (), (), (), f64>,
            >>::terminate(
                &mut sp, &state
            ),
            TerminationStatus::NotTerminated
        );

        sp.intervals.clear(); // Simulate convergence by clearing `intervals`

        // The solver should terminate if there are no more candidate subintervals to search
        assert_eq!(
            <ShubertPiyavskii<f64> as Solver<
                SpTestProblem,
                IterState<f64, (), (), (), (), f64>,
            >>::terminate(
                &mut sp, &state
            ),
            TerminationStatus::Terminated(TerminationReason::SolverConverged)
        );
    }

    /// Validate the solver's transition to the next iteration.
    #[test]
    fn test_next_iter() {
        let mut sp = ShubertPiyavskii::new(MIN_BOUND, MAX_BOUND, LIPSCHITZ_CONST).unwrap();
        let mut problem = Problem::new(SpTestProblem {});
        let (state, _) = sp.init(&mut problem, IterState::new()).unwrap();
        let (state, kv) = sp.next_iter(&mut problem, state).unwrap();

        // The solver should not return metadata
        assert!(kv.is_none());

        let x_sample = sp.sample_point(
            MIN_BOUND,
            MAX_BOUND,
            problem.cost(&MIN_BOUND).unwrap(),
            problem.cost(&MAX_BOUND).unwrap(),
        );

        let x_best = state.param.unwrap();
        let f_best = state.cost;

        let ShubertPiyavskii {
            min_bound,
            max_bound,
            lipschitz_const,
            tolerance,
            intervals,
        } = sp;

        assert_eq!(min_bound.to_ne_bytes(), MIN_BOUND.to_ne_bytes());
        assert_eq!(max_bound.to_ne_bytes(), MAX_BOUND.to_ne_bytes());
        assert_eq!(lipschitz_const.to_ne_bytes(), LIPSCHITZ_CONST.to_ne_bytes());
        assert_eq!(tolerance.to_ne_bytes(), DEFAULT_TOL.to_ne_bytes());
        // One interval created upon initialization, split into two after the first iteration
        assert_eq!(intervals.len(), 2);

        let selected_lower = relative_eq!(x_best, min_bound, epsilon = f64::EPSILON)
            && relative_eq!(
                f_best,
                problem.cost(&min_bound).unwrap(),
                epsilon = f64::EPSILON
            );

        let selected_upper = relative_eq!(x_best, max_bound, epsilon = f64::EPSILON)
            && relative_eq!(
                f_best,
                problem.cost(&max_bound).unwrap(),
                epsilon = f64::EPSILON
            );

        let selected_sample = relative_eq!(x_best, x_sample, epsilon = f64::EPSILON)
            && relative_eq!(
                f_best,
                problem.cost(&x_sample).unwrap(),
                epsilon = f64::EPSILON
            );

        // The current guess should be the lower bound, the upper bound, or the sample point
        assert!(selected_lower || selected_upper || selected_sample);
    }

    /// Validate the solver's ability to convergence to the global minimum.
    #[test]
    fn test_solver() {
        let cost = SpTestProblem {};
        let solver = ShubertPiyavskii::new(MIN_BOUND, MAX_BOUND, LIPSCHITZ_CONST).unwrap();

        let res = Executor::new(cost, solver)
            .configure(|state| state)
            .run()
            .unwrap();

        let state = res.state;
        let mut problem = Problem::new(SpTestProblem {});

        assert!(state.cost - problem.cost(&GLOBAL_MINIMIZER).unwrap() < DEFAULT_TOL);
    }

    /// Validate proper handling of non-finite values (NaN, +Inf, -Inf) at various possible points.
    #[test]
    fn test_nonfinite_errors() {
        test_nan_at_min_bound();
        test_infty_at_max_bound();
        test_neg_infty_at_inner();
    }

    /// Immediate error upon initialization when a non-finite value is attained at the lower bound.
    fn test_nan_at_min_bound() {
        const NAN_MIN_BOUND: f64 = 0.0;
        const NAN_MAX_BOUND: f64 = 1.0;

        #[derive(Clone)]
        struct NaNLowerProblem {}

        impl CostFunction for NaNLowerProblem {
            type Param = f64;
            type Output = f64;

            fn cost(&self, x: &Self::Param) -> Result<Self::Output, Error> {
                match *x {
                    NAN_MIN_BOUND => Ok(f64::NAN),
                    _ => Ok(*x),
                }
            }
        }

        let mut sp = ShubertPiyavskii::new(NAN_MIN_BOUND, NAN_MAX_BOUND, LIPSCHITZ_CONST).unwrap();
        let mut problem = Problem::new(NaNLowerProblem {});
        let res = sp.init(&mut problem, IterState::new());

        assert_error!(
            res,
            ArgminError,
            format!(
                "Invalid parameter: \"{}\"",
                nonfinite_err_msg(NAN_MIN_BOUND)
            )
        );
    }

    // Immediate error upon initialization when a non-finite value is attained at the upper bound.
    fn test_infty_at_max_bound() {
        const INFTY_MIN_BOUND: f64 = 0.0;
        const INFTY_MAX_BOUND: f64 = 1.0;

        #[derive(Clone)]
        struct InftyUpperProblem {}

        impl CostFunction for InftyUpperProblem {
            type Param = f64;
            type Output = f64;

            fn cost(&self, x: &Self::Param) -> Result<Self::Output, Error> {
                match *x {
                    INFTY_MAX_BOUND => Ok(f64::INFINITY),
                    _ => Ok(*x),
                }
            }
        }

        let mut sp =
            ShubertPiyavskii::new(INFTY_MIN_BOUND, INFTY_MAX_BOUND, LIPSCHITZ_CONST).unwrap();
        let mut problem = Problem::new(InftyUpperProblem {});
        let res = sp.init(&mut problem, IterState::new());

        assert_error!(
            res,
            ArgminError,
            format!(
                "Invalid parameter: \"{}\"",
                nonfinite_err_msg(INFTY_MAX_BOUND)
            )
        );
    }

    // Error upon the first iteration when a non-finite value is attained at the sample point.
    fn test_neg_infty_at_inner() {
        const NEG_INFTY_MIN_BOUND: f64 = 0.0;
        const NEG_INFTY_MAX_BOUND: f64 = 1.0;

        #[derive(Clone)]
        struct NegInftyInnerProblem {}

        impl CostFunction for NegInftyInnerProblem {
            type Param = f64;
            type Output = f64;

            fn cost(&self, x: &Self::Param) -> Result<Self::Output, Error> {
                match *x {
                    NEG_INFTY_MIN_BOUND => Ok(*x),
                    NEG_INFTY_MAX_BOUND => Ok(*x),
                    _ => Ok(f64::NEG_INFINITY),
                }
            }
        }

        let mut sp =
            ShubertPiyavskii::new(NEG_INFTY_MIN_BOUND, NEG_INFTY_MAX_BOUND, LIPSCHITZ_CONST)
                .unwrap();
        let mut problem = Problem::new(NegInftyInnerProblem {});
        let (state, _) = sp.init(&mut problem, IterState::new()).unwrap();
        let res = sp.next_iter(&mut problem, state);
        let x_sample = sp.sample_point(
            NEG_INFTY_MIN_BOUND,
            NEG_INFTY_MAX_BOUND,
            problem.cost(&NEG_INFTY_MIN_BOUND).unwrap(),
            problem.cost(&NEG_INFTY_MAX_BOUND).unwrap(),
        );

        assert_error!(
            res,
            ArgminError,
            format!("Invalid parameter: \"{}\"", nonfinite_err_msg(x_sample))
        );
    }
}
