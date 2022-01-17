// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Quasi-Newton methods
//!
//! [BFGS](BFGS/struct.BFGS.html)
//!
//! # References:
//!
//! \[0\] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
//! Springer. ISBN 0-387-30303-0.

pub mod bfgs;
pub mod dfp;
pub mod lbfgs;
pub mod sr1;
pub mod sr1_trustregion;

pub use self::bfgs::*;
pub use self::dfp::*;
pub use self::lbfgs::*;
pub use self::sr1::*;
pub use self::sr1_trustregion::*;
