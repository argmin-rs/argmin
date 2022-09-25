// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Line search methods
//!
//! Line searches are given a position in parameter space and a direction. They obtain a step
//! length in this direction which fulfills a given set of [acceptance conditions](`condition`).
//!
//! These methods are often an integral part of other methods, such as gradient descent.
//! Each algorithm which implements the [`LineSearch`] trait can be used in these optimization
//! methods.
//!
//! ## Available line searches
//!
//! * [Backtracking line search](`BacktrackingLineSearch`)
//! * [More-Thuente line search](`MoreThuenteLineSearch`)
//! * [Hager-Zhang line search](`HagerZhangLineSearch`)
//!
//! ## References
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

mod backtracking;
/// Acceptance conditions
pub mod condition;
mod hagerzhang;
mod morethuente;

pub use self::backtracking::BacktrackingLineSearch;
pub use self::hagerzhang::HagerZhangLineSearch;
pub use self::morethuente::MoreThuenteLineSearch;

/// # Line search trait
///
/// For a method to be used as a line search, it has to implement this trait.
///
/// It enables the optimization method to set the search direction and the initial step length of
/// the line search.
///
/// ## Example
///
/// ```
/// use argmin::solver::linesearch::LineSearch;
///
/// struct MyLineSearch<P, F> {
///     // `P` is the type of the search direction, typically the same as the parameter vector
///     search_direction: P,
///     // `F` is a floating point value (f32 or f64)
///     init_step_length: F,
///     // ...
/// }
///
/// impl<P, F> LineSearch<P, F> for MyLineSearch<P, F> {
///     fn search_direction(&mut self, direction: P) {
///         self.search_direction = direction;
///     }
///
///     fn initial_step_length(&mut self, step_length: F) -> Result<(), argmin::core::Error> {
///         self.init_step_length = step_length;
///         Ok(())
///     }
/// }
/// ```
pub trait LineSearch<P, F> {
    /// Set the search direction
    ///
    /// This is the direction in which the line search will be performed.
    fn search_direction(&mut self, direction: P);

    /// Set the initial step length
    ///
    /// This indicates the first step length which will be tried.
    fn initial_step_length(&mut self, step_length: F) -> Result<(), crate::core::Error>;
}
