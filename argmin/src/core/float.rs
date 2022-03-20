// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::core::{DeserializeOwnedAlias, SerializeAlias};
use num_traits::{Float, FloatConst, FromPrimitive, ToPrimitive};
use std::fmt::{Debug, Display};

/// An alias for float types (`f32`, `f64`) which combines multiple commonly needed traits from
/// `num_traits`, `std::fmt` and for serialization/deserialization (the latter only if the `serde1`
/// feature is enabled). It is automatically implemented for all types which fulfill the trait
/// bounds.
pub trait ArgminFloat:
    'static
    + Float
    + FloatConst
    + FromPrimitive
    + ToPrimitive
    + Debug
    + Display
    + SerializeAlias
    + DeserializeOwnedAlias
{
}

/// `ArgminFloat` is automatically implemented for all types which fulfill the trait bounds.
impl<I> ArgminFloat for I where
    I: 'static
        + Float
        + FloatConst
        + FromPrimitive
        + ToPrimitive
        + Debug
        + Display
        + SerializeAlias
        + DeserializeOwnedAlias
{
}
