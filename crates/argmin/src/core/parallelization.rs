// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

/// Trait alias for `Send`
///
/// If the `rayon` feature is set, it acts as an alias for `Send` and is implemented for all
/// types which implement `Send`. If `rayon` is not set, it will be an "empty" trait
/// implemented for all types.
#[cfg(feature = "rayon")]
pub trait SendAlias: Send {}

/// Trait alias for `Send`
///
/// If the `rayon` feature is set, it acts as an alias for `Send` and is implemented for all
/// types which implement `Send`. If `rayon` is not set, it will be an "empty" trait
/// implemented for all types.
#[cfg(not(feature = "rayon"))]
pub trait SendAlias {}

#[cfg(feature = "rayon")]
impl<T> SendAlias for T where T: Send {}

#[cfg(not(feature = "rayon"))]
impl<T> SendAlias for T {}

/// Trait alias for `Sync`
///
/// If the `rayon` feature is set, it acts as an alias for `Sync` and is implemented for all types
/// which implement `Sync`. If `rayon` is not set, it will be an "empty" trait implemented for all
/// types.
#[cfg(feature = "rayon")]
pub trait SyncAlias: Sync {}

/// Trait alias for `Sync`
///
/// If the `rayon` feature is set, it acts as an alias for `Sync` and is implemented for all types
/// which implement `Sync`. If `rayon` is not set, it will be an "empty" trait implemented for all
/// types.
#[cfg(not(feature = "rayon"))]
pub trait SyncAlias {}

#[cfg(feature = "rayon")]
impl<T> SyncAlias for T where T: Sync {}

#[cfg(not(feature = "rayon"))]
impl<T> SyncAlias for T {}
