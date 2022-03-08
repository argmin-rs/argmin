// Copyright 2018-2022 argmin developers
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
//! \[0\] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
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

use crate::core::SerializeAlias;

/// Common interface for beta update methods (Nonlinear-CG)
pub trait NLCGBetaUpdate<G, P, F>: SerializeAlias {
    /// Update beta
    /// Parameter 1: \nabla f_k
    /// Parameter 2: \nabla f_{k+1}
    /// Parameter 3: p_k
    fn update(&self, nabla_f_k: &G, nabla_f_k_p_1: &G, p_k: &P) -> F;
}
