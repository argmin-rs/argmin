// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! argmin-math provides mathematics related abstractions needed in argmin. It supports
//! implementations of these abstractions for basic `Vec`s and for the `ndarray`, `nalgebra`,
//! and `faer` linear algebra libraries. The traits can of course also be implemented
//! for your own types to make them compatible with argmin.
//!
//! For an introduction on how to use argmin, please also have a look at the
//! [book](https://www.argmin-rs.org/book/).
//!
//! # Usage
//!
//! Add the following line to your dependencies list:
//!
//! ```toml
//! [dependencies]
#![doc = concat!("argmin-math = \"", env!("CARGO_PKG_VERSION"), "\"")]
//! ```
//!
//! This will activate the `primitives` and `vec` features. For other backends see the section
//! below.
//!
//! ## Features
//!
//! Support for the various backends can be switched on via features. Please read this section
//! carefully to the end before choosing a backend.
//!
//! ### Default features
//!
//! | Feature                | Default | Comment                                               |
//! |------------------------|---------|-------------------------------------------------------|
//! | `primitives`           | yes     | basic integer and floating point types                |
//! | `vec`                  | yes     | `Vec`s (basic functionality)                          |
//!
//! ### `ndarray`
//!
//! | Feature                         | Default | Comment                                                            |
//! |---------------------------------|---------|--------------------------------------------------------------------|
//! | `ndarray_latest`                | no      | latest supported version                                           |
//! | `ndarray_latest-nolinalg`       | no      | latest supported version without `ndarray-linalg`                  |
//! | `ndarray_v0_15`                 | no      | version 0.15 with ndarray-linalg 0.16                              |
//! | `ndarray_v0_15-nolinalg`        | no      | version 0.15 without `ndarray-linalg`                              |
//! | `ndarray_v0_14-nolinalg`        | no      | version 0.14 without `ndarray-linalg`                              |
//! | `ndarray_v0_13-nolinalg`        | no      | version 0.13 without `ndarray-linalg`                              |
//!
//! Note that the `*-nolinalg*` features do NOT pull in `ndarray-linalg` as a dependency. This
//! avoids linking against a BLAS library. This will however disable the implementation of
//! `ArgminInv`, meaning that any solver which requires the matrix inverse will not work with the
//! `ndarray` backend. It is recommended to use the `*-nolinalg*` options if the matrix inverse is
//! not needed in order to keep the compilation times low and avoid problems when linking against a
//! BLAS library.
//!
//! Using the `ndarray_*` features with `ndarray-linalg` support may require to explicitly choose
//! the `ndarray-linalg` BLAS backend in your `Cargo.toml` (see the [`ndarray-linalg` documentation
//! for details](https://github.com/rust-ndarray/ndarray-linalg)):
//!
//! ```toml
//! ndarray-linalg = { version = "<appropriate_version>", features = ["<linalg_backend>"] }
//! ```
//!
//! ### `nalgebra`
//!
//! | Feature                | Default | Comment                                  |
//! |------------------------|---------|------------------------------------------|
//! | `nalgebra_latest`      | no      | latest supported version                 |
//! | `nalgebra_v0_33`       | no      | version 0.33                             |
//! | `nalgebra_v0_32`       | no      | version 0.32                             |
//! | `nalgebra_v0_31`       | no      | version 0.31                             |
//! | `nalgebra_v0_30`       | no      | version 0.30                             |
//! | `nalgebra_v0_29`       | no      | version 0.29                             |
//!
//! ### `faer`
//!
//! | Feature                | Default | Comment                                  |
//! |------------------------|---------|------------------------------------------|
//! | `faer_latest`          | no      | latest supported version                 |
//! | `faer_v0_20`           | no      | version 0.20                             |
//!
//! ## Choosing a backend
//!
//! It is not possible to activate two versions of the same backend.
//!
//! The features labeled `*latest*` are an alias for the most recent supported version of the
//! respective backend. It is however recommended to explicitly specify the desired version instead
//! of using any of the `*latest*` features (see section about semantic versioning below).
//!
//! The default features `primitives` and `vec` can be turned off in order to only compile the
//! trait definitions. If another backend is chosen, `primitives` will automatically be turned on
//! again.
//!
//! ### Example
//!
//! Activate support for the latest supported `ndarray` version:
//!
//! ```toml
//! [dependencies]
#![doc = concat!("argmin-math = { version = \"", env!("CARGO_PKG_VERSION"), "\", features = [\"ndarray_latest\"] }")]
//! ```
//!
//! # Semantic versioning
//!
//! This crate follows semantic versioning. Adding a new backend or a new version of a backend is
//! not considered a breaking change. However, your code may still break if you use any of the
//! features containing `*latest*`. It is therefore recommended to specify the actual version of the
//! backend you are using.
//!
//! # Contributing
//!
//! You found a bug? Your favorite backend is not supported? Feel free to open an issue or ideally
//! submit a PR.
//!
//! # License
//!
//! Licensed under either of
//!
//!   * Apache License, Version 2.0,
//!     ([LICENSE-APACHE](https://github.com/argmin-rs/argmin/blob/main/LICENSE-APACHE) or
//!     <http://www.apache.org/licenses/LICENSE-2.0>)
//!   * MIT License ([LICENSE-MIT](https://github.com/argmin-rs/argmin/blob/main/LICENSE-MIT) or
//!     <http://opensource.org/licenses/MIT>)
//!
//! at your option.
//!
//! ## Contribution
//!
//! Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion
//! in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above,
//! without any additional terms or conditions.

#![warn(missing_docs)]
// Explicitly disallow EQ comparison of floats. (This clippy lint is denied by default; however,
// this is just to make sure that it will always stay this way.)
#![deny(clippy::float_cmp)]

cfg_if::cfg_if! {
    if #[cfg(feature = "nalgebra_0_33")] {
        extern crate nalgebra_0_33 as nalgebra;
        trait Allocator<T, R, C = nalgebra::U1>: nalgebra::allocator::Allocator<R, C>
        where
            R: nalgebra::Dim,
            C: nalgebra::Dim,
        {}
        impl<T, R, C, U> Allocator<T, R, C> for U
        where
            U: nalgebra::allocator::Allocator<R, C>,
            R: nalgebra::Dim,
            C: nalgebra::Dim,
        {}
        trait SameShapeAllocator<T, R1, C1, R2, C2>: nalgebra::allocator::SameShapeAllocator<R1, C1, R2, C2>
        where
            R1: nalgebra::Dim,
            C1: nalgebra::Dim,
            R2: nalgebra::Dim,
            C2: nalgebra::Dim,
            nalgebra::constraint::ShapeConstraint: nalgebra::constraint::SameNumberOfRows<R1,R2> + nalgebra::constraint::SameNumberOfColumns<C1,C2>,
        {}
        impl<T, R1, C1, R2, C2, U> SameShapeAllocator<T, R1, C1, R2, C2> for U
        where
            U: nalgebra::allocator::SameShapeAllocator<R1, C1, R2, C2>,
            R1: nalgebra::Dim,
            C1: nalgebra::Dim,
            R2: nalgebra::Dim,
            C2: nalgebra::Dim,
            nalgebra::constraint::ShapeConstraint: nalgebra::constraint::SameNumberOfRows<R1,R2> + nalgebra::constraint::SameNumberOfColumns<C1,C2>,
        {}
        use nalgebra::{
            ClosedAddAssign as ClosedAdd,
            ClosedSubAssign as ClosedSub,
            ClosedDivAssign as ClosedDiv,
            ClosedMulAssign as ClosedMul,
        };
    } else if #[cfg(feature = "nalgebra_0_32")] {
        extern crate nalgebra_0_32 as nalgebra;
        use nalgebra::allocator::{Allocator, SameShapeAllocator};
        use nalgebra::{ClosedAdd, ClosedSub, ClosedDiv, ClosedMul};
    } else if #[cfg(feature = "nalgebra_0_31")] {
        extern crate nalgebra_0_31 as nalgebra;
        use nalgebra::allocator::{Allocator, SameShapeAllocator};
        use nalgebra::{ClosedAdd, ClosedSub, ClosedDiv, ClosedMul};
    } else if #[cfg(feature = "nalgebra_0_30")] {
        extern crate nalgebra_0_30 as nalgebra;
        use nalgebra::allocator::{Allocator, SameShapeAllocator};
        use nalgebra::{ClosedAdd, ClosedSub, ClosedDiv, ClosedMul};
    } else if #[cfg(feature = "nalgebra_0_29")] {
        extern crate nalgebra_0_29 as nalgebra;
        use nalgebra::allocator::{Allocator, SameShapeAllocator};
        use nalgebra::{ClosedAdd, ClosedSub, ClosedDiv, ClosedMul};
    }
}

cfg_if::cfg_if! {
    if #[cfg(feature = "ndarray_0_15")] {
        extern crate ndarray_0_15 as ndarray;
    } else if #[cfg(feature = "ndarray_0_14")]  {
        extern crate ndarray_0_14 as ndarray;
    } else if #[cfg(feature = "ndarray_0_13")]  {
        extern crate ndarray_0_13 as ndarray;
    }
}

cfg_if::cfg_if! {
    if #[cfg(feature = "ndarray-linalg_0_16")] {
        extern crate ndarray_linalg_0_16 as ndarray_linalg;
    }
}

cfg_if::cfg_if! {
    if #[cfg(feature = "num-complex_0_2")] {
        extern crate num_complex_0_2 as num_complex;
    } else if #[cfg(feature = "num-complex_0_3")] {
        extern crate num_complex_0_3 as num_complex;
    } else if #[cfg(feature = "num-complex_0_4")] {
        extern crate num_complex_0_4 as num_complex;
    }
}

cfg_if::cfg_if! {
    if #[cfg(feature = "faer_v0_20")] {
        extern crate faer_0_20 as faer;
    }
}

#[cfg(feature = "primitives")]
mod primitives;
#[cfg(feature = "primitives")]
#[allow(unused_imports)]
pub use crate::primitives::*;

#[cfg(feature = "ndarray_all")]
mod ndarray_m;
#[cfg(feature = "ndarray_all")]
#[allow(unused_imports)]
pub use crate::ndarray_m::*;

#[cfg(feature = "nalgebra_all")]
mod nalgebra_m;
#[cfg(feature = "nalgebra_all")]
#[allow(unused_imports)]
pub use crate::nalgebra_m::*;

#[cfg(feature = "vec")]
mod vec;
#[cfg(feature = "vec")]
#[allow(unused_imports)]
pub use crate::vec::*;

#[cfg(feature = "faer_all")]
mod faer_m;
#[cfg(feature = "faer_all")]
#[allow(unused_imports)]
pub use crate::faer_m::*;

// Re-export of types appearing in the api as recommended here: https://www.lurklurk.org/effective-rust/re-export.html
pub use anyhow::Error;
pub use rand::Rng;

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

/// Compute the l1-norm (`U`) of `self`
pub trait ArgminL1Norm<U> {
    /// Compute the l1-norm (`U`) of `self`
    fn l1_norm(&self) -> U;
}

/// Compute the l2-norm (`U`) of `self`
pub trait ArgminL2Norm<U> {
    /// Compute the l2-norm (`U`) of `self`
    fn l2_norm(&self) -> U;
}

// Sub-optimal: self is moved. ndarray however offers array views...
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
    fn rand_from_range<R: Rng>(min: &Self, max: &Self, rng: &mut R) -> Self;
}

/// Minimum and Maximum of type `T`
pub trait ArgminMinMax {
    /// Select piecewise minimum
    fn min(x: &Self, y: &Self) -> Self;
    /// Select piecewise maximum
    fn max(x: &Self, y: &Self) -> Self;
}

/// Returns a number that represents the sign of `self`.
pub trait ArgminSignum {
    /// Returns a number that represents the sign of `self`.
    fn signum(self) -> Self;
}
