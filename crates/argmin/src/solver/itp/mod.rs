// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # ITP method
//!
//! ## ItpRoot
//!
//! A root-finding algorithm, short for "interpolate, truncate, and project",
//! that achieves superlinear convergence while retaining the worst-case
//! performance of the bisection method.
//!
//! ### References
//!
//! [ITP Method]: https://en.wikipedia.org/wiki/ITP_Method
//! [An Enhancement of the Bisection Method Average Performance Preserving Minmax Optimality]: https://dl.acm.org/doi/10.1145/3423597

mod itp_method;

pub use itp_method::ItpRoot;
