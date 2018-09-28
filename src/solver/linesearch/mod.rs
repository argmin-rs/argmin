// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Argmin Line search methods

/// backtracking algorithm
pub mod backtracking;
/// acceptance conditions
pub mod condition;
/// Hager-Zhang line search algorithm
pub mod hagerzhang;
/// More-Thuente line search algorithm
pub mod morethuente;

pub use self::backtracking::*;
pub use self::condition::*;
pub use self::morethuente::*;
