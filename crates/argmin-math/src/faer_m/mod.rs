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
mod inv;
mod l1norm;
mod l2norm;
mod minmax;
mod mul;
mod random;
//@note(geo-ant):
// scaled addition and subtraction rely on a blanket implementation
// for now and don't need to be tested separately for faer.
// Once specialization becomes stable (or the upstream blanket impl is removed)
// we should re-add these modules.
// mod scaledadd;
// mod scaledsub;
mod signum;
mod sub;
mod transpose;
mod zero;

pub use add::*;
pub use conj::*;
pub use div::*;
pub use dot::*;
pub use eye::*;
pub use inv::*;
pub use l1norm::*;
pub use l2norm::*;
// pub use minmax::*;
// pub use mul::*;
// pub use random::*;
// pub use scaledadd::*;
// pub use scaledsub::*;
// pub use signum::*;
// pub use sub::*;
// pub use transpose::*;
// pub use zero::*;
