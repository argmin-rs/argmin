// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

#[cfg(feature = "serde1")]
use serde::{de::DeserializeOwned, Serialize};

/// Trait alias for `serde`s `Serialize`.
///
/// If the `serde1` feature is set, it acts as an alias for `Serialize` and is implemented for all
/// types which implement `Serialize`. If `serde1` is not set, it will be an "empty" trait
/// implemented for all types.
#[cfg(feature = "serde1")]
pub trait SerializeAlias: Serialize {}

/// Trait alias for `serde`s `Serialize`.
///
/// If the `serde1` feature is set, it acts as an alias for `Serialize` and is implemented for all
/// types which implement `Serialize`. If `serde1` is not set, it will be an "empty" trait
/// implemented for all types.
#[cfg(not(feature = "serde1"))]
pub trait SerializeAlias {}

#[cfg(feature = "serde1")]
impl<T> SerializeAlias for T where T: Serialize {}

#[cfg(not(feature = "serde1"))]
impl<T> SerializeAlias for T {}

/// Trait alias for `serde`s `DeserializeOwned`.
///
/// If the `serde1` feature is set, it acts as an alias for `DeserializeOwned` and is implemented
/// for all types which implement `DeserializeOwned`. If `serde1` is not set, it will be an "empty"
/// trait implemented for all types.
#[cfg(feature = "serde1")]
pub trait DeserializeOwnedAlias: DeserializeOwned {}

/// Trait alias for `serde`s `DeserializeOwned`.
///
/// If the `serde1` feature is set, it acts as an alias for `DeserializeOwned` and is implemented
/// for all types which implement `DeserializeOwned`. If `serde1` is not set, it will be an "empty"
/// trait implemented for all types.
#[cfg(not(feature = "serde1"))]
pub trait DeserializeOwnedAlias {}

#[cfg(feature = "serde1")]
impl<T> DeserializeOwnedAlias for T where T: DeserializeOwned {}

#[cfg(not(feature = "serde1"))]
impl<T> DeserializeOwnedAlias for T {}
