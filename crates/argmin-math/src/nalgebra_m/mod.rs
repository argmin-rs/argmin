// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

#![allow(unused_imports)]

// nalgebra renamed `Dynamic` to `Dyn` in 0.32. Re-export under a single name
// so the per-version backend modules don't need their own cfg_if blocks.
cfg_if::cfg_if! {
    if #[cfg(any(
        feature = "nalgebra_0_29",
        feature = "nalgebra_0_30",
        feature = "nalgebra_0_31",
    ))] {
        pub(crate) use nalgebra::Dynamic as Dyn;
    } else {
        pub(crate) use nalgebra::Dyn;
    }
}

mod add;
mod conj;
mod div;
mod dot;
mod eye;
mod inv;
mod l1norm;
mod l2norm;
mod minmax;
mod mul;
mod permute;
mod pivoted_qr;
#[cfg(feature = "rand")]
mod random;
mod scaledadd;
mod scaledsub;
mod signum;
mod sub;
mod transpose;
mod triangular_solve;
mod zero;

pub use add::*;
pub use conj::*;
pub use div::*;
pub use dot::*;
pub use eye::*;
pub use inv::*;
pub use l1norm::*;
pub use l2norm::*;
pub use minmax::*;
pub use mul::*;
pub use permute::*;
pub use pivoted_qr::*;
#[cfg(feature = "rand")]
pub use random::*;
pub use scaledadd::*;
pub use scaledsub::*;
pub use signum::*;
pub use sub::*;
pub use transpose::*;
pub use triangular_solve::*;
pub use zero::*;
