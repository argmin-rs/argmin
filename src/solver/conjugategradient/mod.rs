// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Conjugate Gradient methods
//!
//! * [Conjugate Gradients](cg/struct.ConjugateGradient.html)
//! * [Nonlinear Conjugate Gradients](nonlinear_cg/struct.NonlinearConjugateGradient.html)
//!
//! # References:
//!
//! [0] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
//! Springer. ISBN 0-387-30303-0.

/// Conjugate gradient method
pub mod cg;

/// Nonlinear conjugate gradient method
pub mod nonlinear_cg;

/// Beta update methods for nonlinear CG
pub mod beta;

pub use self::beta::*;
pub use self::cg::*;
pub use self::nonlinear_cg::*;
