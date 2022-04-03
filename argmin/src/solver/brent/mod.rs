//! Brent's methods
//!
//! # BrentOpt
//!
//! A minimization algorithm combining parabolic interpolation and the
//! golden-section method.  It has the reliability of the golden-section
//! method, but can be faster thanks to the parabolic interpolation steps.
//!
//! ## References
//!
//! "An algorithm with guaranteed convergence for finding a minimum of
//! a function of one variable", _Algorithms for minimization without
//! derivatives_, Richard P. Brent, 1973, Prentice-Hall.
//!
//! # BrentRoot
//!
//! A root-finding algorithm combining the bisection method, the secant method
//! and inverse quadratic interpolation. It has the reliability of bisection
//! but it can be as quick as some of the less-reliable methods.
//!
//! ## References
//!
//! <https://en.wikipedia.org/wiki/Brent%27s_method>

mod brentopt;
mod brentroot;

pub use brentopt::BrentOpt;
pub use brentroot::BrentRoot;
