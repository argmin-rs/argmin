// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Quasi-Newton methods
//!
//! * [`BFGS`]
//! * [`DFP`]
//! * [`LBFGS`]
//! * [`SR1`]
//! * [`SR1TrustRegion`]
//!
//! ## Reference
//!
//! Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
//! Springer. ISBN 0-387-30303-0.

mod bfgs;
mod dfp;
mod lbfgs;
mod sr1;
mod sr1_trustregion;

pub use self::bfgs::BFGS;
pub use self::dfp::DFP;
pub use self::lbfgs::LBFGS;
pub use self::sr1::SR1;
pub use self::sr1_trustregion::SR1TrustRegion;
