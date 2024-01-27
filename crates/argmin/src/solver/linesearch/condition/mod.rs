// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Line search conditions
//!
//! For a step length to be accepted in a line search, it needs to satisfy one of several
//! conditions.
//!
//! A condition exposes an interface defined by the trait
//! [`LineSearchCondition`](`condition::LineSearchCondition`).
//!
//! # Available line search conditions
//!
//! * [`ArmijoCondition`](`condition::ArmijoCondition`)
//! * [`WolfeCondition`](`condition::WolfeCondition`)
//! * [`StrongWolfeCondition`](`condition::StrongWolfeCondition`)
//! * [`GoldsteinCondition`](`condition::GoldsteinCondition`)
//!
//! ## Reference
//!
//! Jorge Nocedal and Stephen J. Wright (2006). Numerical Optimization.
//! Springer. ISBN 0-387-30303-0.

mod armijo;
mod goldstein;
mod strongwolfe;
mod wolfe;

pub use armijo::ArmijoCondition;
pub use goldstein::GoldsteinCondition;
pub use strongwolfe::StrongWolfeCondition;
pub use wolfe::WolfeCondition;

/// Interface which a condition needs to implement.
///
/// # Example
///
/// ```
/// use argmin::solver::linesearch::condition::LineSearchCondition;
///
/// pub struct MyCondition {}
///
/// impl<T, G, F> LineSearchCondition<T, G, F> for MyCondition {
///     fn evaluate_condition(
///         &self,
///         current_cost: F,
///         current_gradient: Option<&G>,
///         initial_cost: F,
///         initial_gradient: &G,
///         search_direction: &T,
///         step_length: F,
///     ) -> bool {
///         // Use the current cost function value, the current gradient, the initial cost function
///         // value, the initial gradient, the search direction and the current step length to
///         // compute whether your condition is met (return `true`) or not (return `false`).
/// #       true
///     }
///
///     fn requires_current_gradient(&self) -> bool {
///         // Indicate to the calling method whether your condition requires the current gradient
///         // (at the position defined by the current step length).
///         true
///     }
/// }
/// ```
pub trait LineSearchCondition<T, G, F> {
    /// Evaluate the condition
    ///
    /// This method has access to the initial cost function value and the initial gradient (at the
    /// initial point from where to search from), the current cost function value at the initial
    /// point plus a step in search direction of length `step_length` as well as (optionally) the
    /// gradient at that point. It further has access to the search direction and the step length.
    ///
    /// It returns `true` if the condition was met and `false` if not.
    fn evaluate_condition(
        &self,
        current_cost: F,
        current_gradient: Option<&G>,
        initial_cost: F,
        initial_gradient: &G,
        search_direction: &T,
        step_length: F,
    ) -> bool;

    /// Indicates whether this condition requires the computation of the gradient at the new point
    ///
    /// This should return `false` if the evaluation of the condition does not require the gradient
    /// at the current point and `true` otherwise.
    fn requires_current_gradient(&self) -> bool;
}
