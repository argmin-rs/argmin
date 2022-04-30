// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Newton Methods
//!
//! * [`Newton`]
//! * [`NewtonCG`]
//!
//! # Reference
//!
//! Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
//! Springer. ISBN 0-387-30303-0.

/// Newton-CG method
mod newton_cg;
/// Newton's method
mod newton_method;

pub use self::newton_cg::NewtonCG;
pub use self::newton_method::Newton;
