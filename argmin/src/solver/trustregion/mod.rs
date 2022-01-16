// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Trust region methods

/// Cauchy Point
pub mod cauchypoint;
/// Dogleg method
pub mod dogleg;
/// Steihaug method
pub mod steihaug;
/// Trust region solver
pub mod trustregion_method;

pub use self::cauchypoint::*;
pub use self::dogleg::*;
pub use self::steihaug::*;
pub use self::trustregion_method::*;

/// Computes reduction ratio
pub fn reduction_ratio<F: crate::core::ArgminFloat>(fxk: F, fxkpk: F, mk0: F, mkpk: F) -> F {
    (fxk - fxkpk) / (mk0 - mkpk)
}
