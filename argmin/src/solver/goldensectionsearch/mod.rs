// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # References:
//!
//! [Wikipedia](https://en.wikipedia.org/wiki/Golden-section_search)

use crate::prelude::*;
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

// Golden ratio is actually 1.61803398874989484820, but that is too much precision for f64.
const GOLDEN_RATIO: f64 = 1.618_033_988_749_895;
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
/// The `min_bound` and `max_bound` arguments define values that bracket the expected minimum.
///
/// # References:
///
/// [Wikipedia](https://en.wikipedia.org/wiki/Golden-section_search)
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
    /// Constructor
    pub fn new(min_bound: F, max_bound: F) -> Self {
        GoldenSectionSearch {
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
        }
    }

    /// Set tolerance
    #[must_use]
    pub fn tolerance(mut self, tol: F) -> Self {
        self.tolerance = tol;
        self
    }
}

impl<O, F> Solver<O> for GoldenSectionSearch<F>
where
    O: ArgminOp<Output = F, Param = F, Float = F>,
    F: ArgminFloat,
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
                (init_estimate, init_estimate + self.g2 * max_ie)
            } else {
                (init_estimate - self.g2 * ie_min, init_estimate)
            };
            self.x1 = x1;
            self.x2 = x2;
            self.f1 = op.apply(&self.x1)?;
            self.f2 = op.apply(&self.x2)?;
            if self.f1 < self.f2 {
                Ok(Some(ArgminIterData::new().param(self.x1).cost(self.f1)))
            } else {
                Ok(Some(ArgminIterData::new().param(self.x2).cost(self.f2)))
            }
        }
    }

    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &IterState<O>,
    ) -> Result<ArgminIterData<O>, Error> {
        if self.tolerance * (self.x1.abs() + self.x2.abs()) >= (self.x3 - self.x0).abs() {
            return Ok(ArgminIterData::new()
                .param(state.param)
                .cost(state.cost)
                .termination_reason(TerminationReason::TargetToleranceReached));
        }

        if self.f2 < self.f1 {
            self.x0 = self.x1;
            self.x1 = self.x2;
            self.x2 = self.g1 * self.x1 + self.g2 * self.x3;
            self.f1 = self.f2;
            self.f2 = op.apply(&self.x2)?;
        } else {
            self.x3 = self.x2;
            self.x2 = self.x1;
            self.x1 = self.g1 * self.x2 + self.g2 * self.x0;
            self.f2 = self.f1;
            self.f1 = op.apply(&self.x1)?;
        }
        if self.f1 < self.f2 {
            Ok(ArgminIterData::new().param(self.x1).cost(self.f1))
        } else {
            Ok(ArgminIterData::new().param(self.x2).cost(self.f2))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_trait_impl;

    test_trait_impl!(golden_section_search, GoldenSectionSearch<f64>);
}
