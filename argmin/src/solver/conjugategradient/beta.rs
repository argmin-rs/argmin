// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Beta update methods
//!
//! TODO: Proper documentation.
//!
//! # References:
//!
//! \[0\] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
//! Springer. ISBN 0-387-30303-0.

use crate::core::{ArgminFloat, ArgminNLCGBetaUpdate};
use argmin_math::{ArgminDot, ArgminNorm, ArgminSub};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// Fletcher and Reeves (FR) method
/// TODO: Reference
#[derive(Default, Copy, Clone, Debug)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct FletcherReeves {}

impl FletcherReeves {
    /// Constructor
    pub fn new() -> Self {
        FletcherReeves {}
    }
}

impl<T, F> ArgminNLCGBetaUpdate<T, F> for FletcherReeves
where
    T: Clone + ArgminDot<T, F>,
    F: ArgminFloat,
{
    fn update(&self, dfk: &T, dfk1: &T, _pk: &T) -> F {
        dfk1.dot(dfk1) / dfk.dot(dfk)
    }
}

/// Polak and Ribiere (PR) method
/// TODO: Reference
#[derive(Default, Copy, Clone, Debug)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct PolakRibiere {}

impl PolakRibiere {
    /// Constructor
    pub fn new() -> Self {
        PolakRibiere {}
    }
}

impl<T, F> ArgminNLCGBetaUpdate<T, F> for PolakRibiere
where
    T: ArgminDot<T, F> + ArgminSub<T, T> + ArgminNorm<F>,
    F: ArgminFloat,
{
    fn update(&self, dfk: &T, dfk1: &T, _pk: &T) -> F {
        let dfk_norm_sq = dfk.norm().powi(2);
        dfk1.dot(&dfk1.sub(dfk)) / dfk_norm_sq
    }
}

/// Polak and Ribiere Plus (PR+) method
/// TODO: Reference
#[derive(Default, Copy, Clone, Debug)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct PolakRibierePlus {}

impl PolakRibierePlus {
    /// Constructor
    pub fn new() -> Self {
        PolakRibierePlus {}
    }
}

impl<T, F> ArgminNLCGBetaUpdate<T, F> for PolakRibierePlus
where
    T: ArgminDot<T, F> + ArgminSub<T, T> + ArgminNorm<F>,
    F: ArgminFloat,
{
    fn update(&self, dfk: &T, dfk1: &T, _pk: &T) -> F {
        let dfk_norm_sq = dfk.norm().powi(2);
        let beta = dfk1.dot(&dfk1.sub(dfk)) / dfk_norm_sq;
        F::from_f64(0.0).unwrap().max(beta)
    }
}

/// Hestenes and Stiefel (HS) method
/// TODO: Reference
#[derive(Default, Copy, Clone, Debug)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct HestenesStiefel {}

impl HestenesStiefel {
    /// Constructor
    pub fn new() -> Self {
        HestenesStiefel {}
    }
}

impl<T, F> ArgminNLCGBetaUpdate<T, F> for HestenesStiefel
where
    T: ArgminDot<T, F> + ArgminSub<T, T>,
    F: ArgminFloat,
{
    fn update(&self, dfk: &T, dfk1: &T, pk: &T) -> F {
        let d = dfk1.sub(dfk);
        dfk1.dot(&d) / d.dot(pk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_trait_impl;

    test_trait_impl!(fletcher_reeves, FletcherReeves);
    test_trait_impl!(polak_ribiere, PolakRibiere);
    test_trait_impl!(polak_ribiere_plus, PolakRibierePlus);
    test_trait_impl!(hestenes_stiefel, HestenesStiefel);
}
