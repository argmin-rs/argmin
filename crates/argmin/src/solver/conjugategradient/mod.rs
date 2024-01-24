// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Conjugate Gradient methods
//!
//! * [Conjugate Gradient](`ConjugateGradient`)
//! * [Nonlinear Conjugate Gradient](`NonlinearConjugateGradient`)
//!
//! ## Reference
//!
//! Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
//! Springer. ISBN 0-387-30303-0.

mod cg;
mod nonlinear_cg;

pub mod beta;

pub use self::cg::ConjugateGradient;
pub use self::nonlinear_cg::NonlinearConjugateGradient;
