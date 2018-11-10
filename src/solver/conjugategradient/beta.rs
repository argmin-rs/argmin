// Copyright 2018 Stefan Kroboth
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
//! [0] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
//! Springer. ISBN 0-387-30303-0.
// //!
// //! # Example
// //!
// //! ```rust
// //! todo
// //! ```

use prelude::*;

/// Fletcher and Reeves (FR) method
/// TODO: Reference
pub struct FletcherReeves {}

impl FletcherReeves {
    /// Constructor
    pub fn new() -> Self {
        FletcherReeves {}
    }
}

impl<T> ArgminNLCGBetaUpdate<T> for FletcherReeves
where
    T: Clone + ArgminDot<T, f64>,
{
    fn update(&self, dfk: &T, dfk1: &T, _pk: &T) -> f64 {
        dfk1.dot(dfk1.clone()) / dfk.dot(dfk.clone())
    }
}

/// Polak and Ribiere (PR) method
/// TODO: Reference
pub struct PolakRibiere {}

impl PolakRibiere {
    /// Constructor
    pub fn new() -> Self {
        PolakRibiere {}
    }
}

impl<T> ArgminNLCGBetaUpdate<T> for PolakRibiere
where
    T: Clone + ArgminDot<T, f64> + ArgminSub<T> + ArgminNorm<f64>,
{
    fn update(&self, dfk: &T, dfk1: &T, _pk: &T) -> f64 {
        let dfk_norm_sq = dfk.norm().powi(2);
        dfk1.dot(dfk1.sub(dfk.clone())) / dfk_norm_sq
    }
}

/// Polak and Ribiere Plus (PR+) method
/// TODO: Reference
pub struct PolakRibierePlus {}

impl PolakRibierePlus {
    /// Constructor
    pub fn new() -> Self {
        PolakRibierePlus {}
    }
}

impl<T> ArgminNLCGBetaUpdate<T> for PolakRibierePlus
where
    T: Clone + ArgminDot<T, f64> + ArgminSub<T> + ArgminNorm<f64>,
{
    fn update(&self, dfk: &T, dfk1: &T, _pk: &T) -> f64 {
        let dfk_norm_sq = dfk.norm().powi(2);
        let beta = dfk1.dot(dfk1.sub(dfk.clone())) / dfk_norm_sq;
        0.0f64.max(beta)
    }
}

/// Hestenes and Stiefel (HS) method
/// TODO: Reference
pub struct HestenesStiefel {}

impl HestenesStiefel {
    /// Constructor
    pub fn new() -> Self {
        HestenesStiefel {}
    }
}

impl<T> ArgminNLCGBetaUpdate<T> for HestenesStiefel
where
    T: Clone + ArgminDot<T, f64> + ArgminSub<T> + ArgminNorm<f64>,
{
    fn update(&self, dfk: &T, dfk1: &T, pk: &T) -> f64 {
        let d = dfk1.sub(dfk.clone());
        dfk1.dot(d.clone()) / d.dot(pk.clone())
    }
}
