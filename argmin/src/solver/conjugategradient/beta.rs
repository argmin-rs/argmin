// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Beta update methods for [`NonlinearConjugateGradient`](`crate::solver::conjugategradient::NonlinearConjugateGradient`)
//!
//! These methods define the update procedure for
//! [`NonlinearConjugateGradient`](`crate::solver::conjugategradient::NonlinearConjugateGradient`).
//! They are based on the [`NLCGBetaUpdate`] trait which enables users to implement their own beta
//! update methods.
//!
//! # Reference
//!
//! \[0\] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
//! Springer. ISBN 0-387-30303-0.

use crate::core::{ArgminFloat, SerializeAlias};
use argmin_math::{ArgminDot, ArgminNorm, ArgminSub};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// Interface for beta update methods ([`NonlinearConjugateGradient`](`crate::solver::conjugategradient::NonlinearConjugateGradient`))
///
/// # Example
///
/// ```
/// # use argmin::core::{ArgminFloat, NLCGBetaUpdate};
/// #[cfg(feature = "serde1")]
/// use serde::{Deserialize, Serialize};
///
/// #[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
/// struct MyBetaMethod {}
///
/// impl<G, P, F> NLCGBetaUpdate<G, P, F> for MyBetaMethod
/// where
///     F: ArgminFloat,
/// {
///     fn update(&self, dfk: &G, dfk1: &G, p_k: &P) -> F {
///         // Compute updated beta
/// #       F::nan()
///     }
/// }
/// ```
pub trait NLCGBetaUpdate<G, P, F>: SerializeAlias {
    /// Update beta.
    ///
    /// # Parameters
    ///
    /// * `\nabla f_k`
    /// * `\nabla f_{k+1}`
    /// * `p_k`
    fn update(&self, nabla_f_k: &G, nabla_f_k_p_1: &G, p_k: &P) -> F;
}

/// Fletcher and Reeves (FR) method
///
/// Formula: `<\nabla f_{k+1}, \nabla f_{k+1}> / <\nabla f_k, \nabla f_k>`
#[derive(Default, Copy, Clone, Debug)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct FletcherReeves {}

impl FletcherReeves {
    /// Construct a new instance of `FletcherReeves`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::conjugategradient::beta::FletcherReeves;
    /// let beta_method = FletcherReeves::new();
    /// ```
    pub fn new() -> Self {
        FletcherReeves {}
    }
}

impl<G, P, F> NLCGBetaUpdate<G, P, F> for FletcherReeves
where
    G: ArgminDot<G, F>,
    F: ArgminFloat,
{
    /// Update beta using the Fletcher-Reeves method.
    ///
    /// Formula: `<\nabla f_{k+1}, \nabla f_{k+1}> / <\nabla f_k, \nabla f_k>`
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate approx;
    /// # use approx::assert_relative_eq;
    /// # use argmin::solver::conjugategradient::beta::{NLCGBetaUpdate, FletcherReeves};
    /// # let dfk = vec![1f64, 2.0];
    /// # let dfk1 = vec![3f64, 4.0];
    /// let beta_method = FletcherReeves::new();
    /// let beta: f64 = beta_method.update(&dfk, &dfk1, &());
    /// # assert_relative_eq!(beta, 5.0, epsilon = f64::EPSILON);
    /// ```
    fn update(&self, dfk: &G, dfk1: &G, _pk: &P) -> F {
        dfk1.dot(dfk1) / dfk.dot(dfk)
    }
}

/// Polak and Ribiere (PR) method
#[derive(Default, Copy, Clone, Debug)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct PolakRibiere {}

impl PolakRibiere {
    /// Construct a new instance of `PolakRibiere`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::conjugategradient::beta::PolakRibiere;
    /// let beta_method = PolakRibiere::new();
    /// ```
    pub fn new() -> Self {
        PolakRibiere {}
    }
}

impl<G, P, F> NLCGBetaUpdate<G, P, F> for PolakRibiere
where
    G: ArgminDot<G, F> + ArgminSub<G, G> + ArgminNorm<F>,
    F: ArgminFloat,
{
    /// Update beta using the Polak-Ribiere method.
    ///
    /// Formula: `<\nabla f_{k+1}, (\nabla f_{k+1} - \nabla f_k)> / ||\nabla f_k||^2`
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate approx;
    /// # use approx::assert_relative_eq;
    /// # use argmin::solver::conjugategradient::beta::{NLCGBetaUpdate, PolakRibiere};
    /// # let dfk = vec![1f64, 2.0];
    /// # let dfk1 = vec![3f64, 4.0];
    /// let beta_method = PolakRibiere::new();
    /// let beta = beta_method.update(&dfk, &dfk1, &());
    /// # assert_relative_eq!(beta, 14.0/5.0, epsilon = f64::EPSILON);
    /// ```
    fn update(&self, dfk: &G, dfk1: &G, _pk: &P) -> F {
        let dfk_norm_sq = dfk.norm().powi(2);
        dfk1.dot(&dfk1.sub(dfk)) / dfk_norm_sq
    }
}

/// Polak and Ribiere Plus (PR+) method
///
/// Formula: `max(0, <\nabla f_{k+1}, (\nabla f_{k+1} - \nabla f_k)> / ||\nabla f_k||^2)`
#[derive(Default, Copy, Clone, Debug)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct PolakRibierePlus {}

impl PolakRibierePlus {
    /// Construct a new instance of `PolakRibierePlus`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::conjugategradient::beta::PolakRibierePlus;
    /// let beta_method = PolakRibierePlus::new();
    /// ```
    pub fn new() -> Self {
        PolakRibierePlus {}
    }
}

impl<G, P, F> NLCGBetaUpdate<G, P, F> for PolakRibierePlus
where
    G: ArgminDot<G, F> + ArgminSub<G, G> + ArgminNorm<F>,
    F: ArgminFloat,
{
    /// Update beta using the Polak-Ribiere+ (PR+) method.
    ///
    /// Formula: `max(0, <\nabla f_{k+1}, (\nabla f_{k+1} - \nabla f_k)> / ||\nabla f_k||^2)`
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate approx;
    /// # use approx::assert_relative_eq;
    /// # use argmin::solver::conjugategradient::beta::{NLCGBetaUpdate, PolakRibierePlus};
    /// # let dfk = vec![1f64, 2.0];
    /// # let dfk1 = vec![3f64, 4.0];
    /// let beta_method = PolakRibierePlus::new();
    /// let beta = beta_method.update(&dfk, &dfk1, &());
    /// # assert_relative_eq!(beta, 14.0/5.0, epsilon = f64::EPSILON);
    /// #
    /// # let dfk = vec![5f64, 6.0];
    /// # let dfk1 = vec![3f64, 4.0];
    /// # let beta_method = PolakRibierePlus::new();
    /// # let beta = beta_method.update(&dfk, &dfk1, &());
    /// # assert_relative_eq!(beta, 0.0, epsilon = f64::EPSILON);
    /// ```
    fn update(&self, dfk: &G, dfk1: &G, _pk: &P) -> F {
        let dfk_norm_sq = dfk.norm().powi(2);
        let beta = dfk1.dot(&dfk1.sub(dfk)) / dfk_norm_sq;
        F::from_f64(0.0).unwrap().max(beta)
    }
}

/// Hestenes and Stiefel (HS) method
///
/// Formula: `<\nabla f_{k+1}, (\nabla f_{k+1} - \nabla f_k)> / <(\nabla f_{k+1} - \nabla f_k), p_k>`
#[derive(Default, Copy, Clone, Debug)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct HestenesStiefel {}

impl HestenesStiefel {
    /// Construct a new instance of `HestenesStiefel`.
    ///
    /// # Example
    ///
    /// ```
    /// # use argmin::solver::conjugategradient::beta::HestenesStiefel;
    /// let beta_method = HestenesStiefel::new();
    /// ```
    pub fn new() -> Self {
        HestenesStiefel {}
    }
}

impl<G, P, F> NLCGBetaUpdate<G, P, F> for HestenesStiefel
where
    G: ArgminDot<G, F> + ArgminDot<P, F> + ArgminSub<G, G>,
    F: ArgminFloat,
{
    /// Update beta using the Hestenes-Stiefel method.
    ///
    /// Formula: `<\nabla f_{k+1}, (\nabla f_{k+1} - \nabla f_k)> / <(\nabla f_{k+1} - \nabla f_k), p_k>`
    ///
    /// # Example
    ///
    /// ```
    /// # extern crate approx;
    /// # use approx::assert_relative_eq;
    /// # use argmin::solver::conjugategradient::beta::{NLCGBetaUpdate, HestenesStiefel};
    /// # let dfk = vec![1f64, 2.0];
    /// # let dfk1 = vec![3f64, 4.0];
    /// # let pk = vec![5f64, 6.0];
    /// let beta_method = HestenesStiefel::new();
    /// let beta: f64 = beta_method.update(&dfk, &dfk1, &pk);
    /// # assert_relative_eq!(beta, 14.0/22.0, epsilon = f64::EPSILON);
    /// ```
    fn update(&self, dfk: &G, dfk1: &G, pk: &P) -> F {
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
