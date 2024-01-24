// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Trust region method
//!
//! The trust region method approximates the cost function within a certain region around the
//! current point in parameter space. Depending on the quality of this approximation, the region is
//! either expanded or contracted.
//!
//! For more details see [`TrustRegion`].
//!
//! ## Reference
//!
//! Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
//! Springer. ISBN 0-387-30303-0.

/// Cauchy Point
mod cauchypoint;
/// Dogleg method
mod dogleg;
/// Steihaug method
mod steihaug;
/// Trust region solver
mod trustregion_method;

pub use self::cauchypoint::*;
pub use self::dogleg::*;
pub use self::steihaug::*;
pub use self::trustregion_method::*;

/// An interface methods which calculate approximate steps for trust region methods must implement.
///
/// # Example
///
/// ```
/// use argmin::solver::trustregion::TrustRegionRadius;
///
/// struct MySubProblem<F> {
///     radius: F
/// }
///
/// impl<F> TrustRegionRadius<F> for MySubProblem<F> {
///     fn set_radius(&mut self, radius: F) {
///         self.radius = radius
///     }
/// }
/// ```
pub trait TrustRegionRadius<F> {
    /// Set the initial radius
    ///
    /// # Example
    ///
    /// ```
    /// use argmin::solver::trustregion::TrustRegionRadius;
    /// # use argmin::core::ArgminFloat;
    ///
    /// # struct MySubProblem<F> {
    /// #     radius: F
    /// # }
    /// #
    /// # impl<F: ArgminFloat> MySubProblem<F> {
    /// #     pub fn new() -> Self {
    /// #         MySubProblem { radius: F::from_f64(1.0f64).unwrap() }
    /// #     }
    /// # }
    /// #
    /// # impl<F> TrustRegionRadius<F> for MySubProblem<F> {
    /// #     fn set_radius(&mut self, radius: F) {
    /// #         self.radius = radius
    /// #     }
    /// # }
    /// let mut subproblem = MySubProblem::new();
    ///
    /// subproblem.set_radius(0.8);
    /// ```
    fn set_radius(&mut self, radius: F);
}

/// Computes reduction ratio
pub fn reduction_ratio<F: crate::core::ArgminFloat>(fxk: F, fxkpk: F, mk0: F, mkpk: F) -> F {
    (fxk - fxkpk) / (mk0 - mkpk)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_reduction_ration() {
        let fxk = 10.0f64;
        let fxkpk = 6.0;
        let mk0 = 12.0;
        let mkpk = 10.0;

        assert_relative_eq!(
            reduction_ratio(fxk, fxkpk, mk0, mkpk),
            2.0f64,
            epsilon = std::f64::EPSILON
        );
    }
}
