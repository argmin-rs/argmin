// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

#![allow(unused_imports)]

mod add;
mod conj;
mod div;
mod dot;
mod eye;
#[cfg(any(feature = "ndarray-linalg_0_16", feature = "ndarray-linalg_0_17",))]
mod inv;
mod l1norm;
mod l2norm;
mod minmax;
mod mul;
#[cfg(feature = "rand")]
mod random;
mod scaledadd;
mod scaledsub;
mod signum;
mod sub;
mod transpose;
mod zero;

pub use add::*;
pub use conj::*;
pub use div::*;
pub use dot::*;
pub use eye::*;
#[cfg(any(feature = "ndarray-linalg_0_16", feature = "ndarray-linalg_0_17",))]
pub use inv::*;
pub use l1norm::*;
pub use l2norm::*;
pub use minmax::*;
pub use mul::*;
pub use scaledadd::*;
pub use scaledsub::*;
pub use signum::*;
pub use sub::*;
pub use transpose::*;
pub use zero::*;
