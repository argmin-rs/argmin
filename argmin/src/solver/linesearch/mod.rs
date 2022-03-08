// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Line search methods
//!
//! * [Backtracking line search](backtracking/struct.BacktrackingLineSearch.html)
//! * [More-Thuente line search](morethuente/struct.MoreThuenteLineSearch.html)
//! * [Hager-Zhang line search](hagerzhang/struct.HagerZhangLineSearch.html)
//!
//! # References:
//!
//! \[0\] Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
//! Springer. ISBN 0-387-30303-0.
//!
//! \[1\] Jorge J. More and David J. Thuente. "Line search algorithms with guaranteed sufficient
//! decrease." ACM Trans. Math. Softw. 20, 3 (September 1994), 286-307.
//! DOI: <https://doi.org/10.1145/192115.192132>
//!
//! \[2\] William W. Hager and Hongchao Zhang. "A new conjugate gradient method with guaranteed
//! descent and an efficient line search." SIAM J. Optim. 16(1), 2006, 170-192.
//! DOI: <https://doi.org/10.1137/030601880>

/// Backtracking line search algorithm
pub mod backtracking;
/// Acceptance conditions
pub mod condition;
/// Hager-Zhang line search algorithm
pub mod hagerzhang;
/// More-Thuente line search algorithm
pub mod morethuente;

pub use self::backtracking::*;
pub use self::condition::*;
pub use self::hagerzhang::*;
pub use self::morethuente::*;

use crate::core::Error;

/// Defines a common interface for line search methods.
pub trait LineSearch<P, F> {
    /// Set the search direction
    fn set_search_direction(&mut self, direction: P);

    /// Set the initial step length
    fn set_init_alpha(&mut self, step_length: F) -> Result<(), Error>;
}
