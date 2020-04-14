// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # References:
//!
//! [Wikipedia](https://en.wikipedia.org/wiki/Golden-section_search)

use crate::prelude::*;
use serde::{Deserialize, Serialize};

const GOLDEN_RATIO: f64 = 1.61803398874989484820;
const G1: f64 = -1.0 + GOLDEN_RATIO;
const G2: f64 = 1.0 - G1;

/// Golden-section search
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
/// The `min_bound` and `max_bound` arguments define values that bracket the expected minimum. The
/// `init_estimate` argument is the initial estimate of the minimum that is required to be larger
/// than `min_bound` and smaller than `max_bound`.
///
/// # References:
///
/// [Wikipedia](https://en.wikipedia.org/wiki/Golden-section_search)
#[derive(Clone, Serialize, Deserialize)]
pub struct GoldenSectionSearch<O: ArgminOp> {
    min_bound: f64,
    max_bound: f64,
    init_estimate: f64,
    tolerance: f64,

    x0: f64,
    x1: f64,
    x2: f64,
    x3: f64,
    f1: O::Output,
    f2: O::Output,
}

impl<O> GoldenSectionSearch<O>
where
    O: ArgminOp,
    O::Output: ArgminZero,
{
    /// Constructor
    pub fn new(min_bound: f64, max_bound: f64) -> Self {
        GoldenSectionSearch {
            min_bound,
            max_bound,
            init_estimate: 0.0,
            tolerance: 0.01,
            x0: min_bound,
            x1: 0.0,
            x2: 0.0,
            x3: max_bound,
            f1: O::Output::zero(),
            f2: O::Output::zero(),
        }
    }

    /// Set tolerance
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }
}

impl<O> Solver<O> for GoldenSectionSearch<O>
where
    O: ArgminOp<Param = f64, Output = f64>,
{
    const NAME: &'static str = "Golden-section search";

    fn init(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &IterState<O>,
    ) -> Result<Option<ArgminIterData<O>>, Error> {
        let init_estimate = state.param;
        if init_estimate < self.min_bound || init_estimate > self.max_bound {
            Err(ArgminError::InvalidParameter {
                text: "Initial estimate must be ∈ [min_bound, max_bound].".to_string(),
            }
            .into())
        } else {
            let ie_min = init_estimate - self.min_bound;
            let max_ie = self.max_bound - init_estimate;
            let (x1, x2) = if max_ie.abs() > ie_min.abs() {
                (init_estimate, init_estimate + G2 * max_ie)
            } else {
                (init_estimate - G2 * ie_min, init_estimate)
            };
            self.x1 = x1;
            self.x2 = x2;
            self.f1 = op.apply(&self.x1)?;
            self.f2 = op.apply(&self.x2)?;
            Ok(Some(ArgminIterData::new()))
        }
    }

    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        _state: &IterState<O>,
    ) -> Result<ArgminIterData<O>, Error> {
        if self.tolerance * (self.x1.abs() + self.x2.abs()) >= (self.x3 - self.x0).abs() {
            return Ok(ArgminIterData::new()
                .param(if self.f1 < self.f2 { self.x1 } else { self.x2 })
                .termination_reason(TerminationReason::TargetToleranceReached));
        }

        if self.f2 < self.f1 {
            self.x0 = self.x1;
            self.x1 = self.x2;
            self.x2 = G1 * self.x1 + G2 * self.x3;
            self.f1 = self.f2;
            self.f2 = op.apply(&self.x2)?;
        } else {
            self.x3 = self.x2;
            self.x2 = self.x1;
            self.x1 = G1 * self.x2 + G2 * self.x0;
            self.f2 = self.f1;
            self.f1 = op.apply(&self.x1)?;
        }
        Ok(ArgminIterData::new())
    }
}
