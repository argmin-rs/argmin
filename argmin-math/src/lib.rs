// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Math
//!
//! Mathematics related traits which some solvers require. This provides an abstraction over
//! different types of parameter vectors. The idea is, that it does not matter whether you would
//! like to use simple `Vec`s, `ndarray`, `nalgebra` or custom defined types: As long as the traits
//! required by the solver are implemented, you should be fine. In this module several of these
//! traits are defined and implemented. These will be extended as needed. They are also already
//! implemented for basic `Vec`s, and will in the future also be implemented for types defined by
//! `ndarray` and `nalgebra`.
//!
//! # TODO
//!
//! * Implement tests for Complex<T> impls

#[cfg(feature = "nalgebra_v0_29")]
extern crate nalgebra_0_29 as nalgebra;
#[cfg(feature = "nalgebra_v0_30")]
extern crate nalgebra_0_30 as nalgebra;

#[cfg(feature = "ndarray_v0_13")]
extern crate ndarray_0_13 as ndarray;
#[cfg(feature = "ndarray_v0_14")]
extern crate ndarray_0_14 as ndarray;
#[cfg(feature = "ndarray_v0_15")]
extern crate ndarray_0_15 as ndarray;

#[cfg(feature = "ndarray-linalg_0_12")]
extern crate ndarray_linalg_0_12 as ndarray_linalg;
#[cfg(feature = "ndarray-linalg_0_13")]
extern crate ndarray_linalg_0_13 as ndarray_linalg;
#[cfg(feature = "ndarray-linalg_0_14")]
extern crate ndarray_linalg_0_14 as ndarray_linalg;

#[cfg(feature = "num-complex_0_2")]
extern crate num_complex_0_2 as num_complex;
#[cfg(all(feature = "num-complex_0_3", not(feature = "num-complex_0_2")))]
extern crate num_complex_0_3 as num_complex;
#[cfg(all(
    feature = "num-complex_0_4",
    not(feature = "num-complex_0_3"),
    not(feature = "num-complex_0_2")
))]
extern crate num_complex_0_4 as num_complex;

#[cfg(feature = "primitives")]
mod primitives;
#[cfg(feature = "primitives")]
pub use crate::primitives::*;

#[cfg(feature = "ndarray_all")]
mod ndarray_m;
#[cfg(feature = "ndarray_all")]
pub use crate::ndarray_m::*;

#[cfg(feature = "nalgebra_all")]
mod nalgebra_m;
#[cfg(feature = "nalgebra_all")]
pub use crate::nalgebra_m::*;

#[cfg(feature = "vec")]
mod vec;
#[cfg(feature = "vec")]
pub use crate::vec::*;

use anyhow::Error;

/// Dot/scalar product of `T` and `self`
pub trait ArgminDot<T, U> {
    /// Dot/scalar product of `T` and `self`
    fn dot(&self, other: &T) -> U;
}

/// Dot/scalar product of `T` and `self` weighted by W (p^TWv)
pub trait ArgminWeightedDot<T, U, V> {
    /// Dot/scalar product of `T` and `self`
    fn weighted_dot(&self, w: &V, vec: &T) -> U;
}

/// Return param vector of all zeros (for now, this is a hack. It should be done better)
pub trait ArgminZero {
    /// Return zero(s)
    fn zero() -> Self;
}

/// Return the conjugate
pub trait ArgminConj {
    /// Return conjugate
    #[must_use]
    fn conj(&self) -> Self;
}

/// Zero for dynamically sized objects
pub trait ArgminZeroLike {
    /// Return zero(s)
    #[must_use]
    fn zero_like(&self) -> Self;
}

/// Identity matrix
pub trait ArgminEye {
    /// Identity matrix of size `n`
    fn eye(n: usize) -> Self;
    /// Identity matrix of same size as `self`
    #[must_use]
    fn eye_like(&self) -> Self;
}

/// Add a `T` to `self`
pub trait ArgminAdd<T, U> {
    /// Add a `T` to `self`
    fn add(&self, other: &T) -> U;
}

/// Subtract a `T` from `self`
pub trait ArgminSub<T, U> {
    /// Subtract a `T` from `self`
    fn sub(&self, other: &T) -> U;
}

/// (Pointwise) Multiply a `T` with `self`
pub trait ArgminMul<T, U> {
    /// (Pointwise) Multiply a `T` with `self`
    fn mul(&self, other: &T) -> U;
}

/// (Pointwise) Divide a `T` by `self`
pub trait ArgminDiv<T, U> {
    /// (Pointwise) Divide a `T` by `self`
    fn div(&self, other: &T) -> U;
}

/// Add a `T` scaled by an `U` to `self`
pub trait ArgminScaledAdd<T, U, V> {
    /// Add a `T` scaled by an `U` to `self`
    fn scaled_add(&self, factor: &U, vec: &T) -> V;
}

/// Subtract a `T` scaled by an `U` from `self`
pub trait ArgminScaledSub<T, U, V> {
    /// Subtract a `T` scaled by an `U` from `self`
    fn scaled_sub(&self, factor: &U, vec: &T) -> V;
}

/// Compute the l2-norm (`U`) of `self`
pub trait ArgminNorm<U> {
    /// Compute the l2-norm (`U`) of `self`
    fn norm(&self) -> U;
}

// Suboptimal: self is moved. ndarray however offers array views...
/// Return the transpose (`U`) of `self`
pub trait ArgminTranspose<U> {
    /// Transpose
    fn t(self) -> U;
}

/// Compute the inverse (`T`) of `self`
pub trait ArgminInv<T> {
    /// Compute the inverse
    fn inv(&self) -> Result<T, Error>;
}

/// Create a random number
pub trait ArgminRandom {
    /// Get a random element between min and max,
    fn rand_from_range(min: &Self, max: &Self) -> Self;
}

/// Minimum and Maximum of type `T`
pub trait ArgminMinMax {
    /// Select piecewise minimum
    fn min(x: &Self, y: &Self) -> Self;
    /// Select piecewise maximum
    fn max(x: &Self, y: &Self) -> Self;
}
