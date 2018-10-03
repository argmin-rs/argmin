// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Optimizaton toolbox
//!
//! TODOs

#![warn(missing_docs)]
#![feature(custom_attribute)]
#![feature(unrestricted_attribute_tokens)]
#![allow(unused_attributes)]

#[macro_use]
extern crate argmin_core;
#[macro_use]
extern crate argmin_codegen;
extern crate argmin_testfunctions;
extern crate rand;

/// Definition of all relevant traits
pub mod prelude;

/// solvers
pub mod solver;

use argmin_core::*;

/// testfunctions
pub mod testfunctions {
    pub use argmin_testfunctions::*;
}
