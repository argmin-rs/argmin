// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Newton Methods
//!
//! # References:
//!
//! [0] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
//! Springer. ISBN 0-387-30303-0.

/// Gauss-Newton method
pub mod gaussnewton;
/// Gauss-Newton method with linesearch
pub mod gaussnewton_linesearch;

pub use self::gaussnewton::*;
pub use self::gaussnewton_linesearch::*;
