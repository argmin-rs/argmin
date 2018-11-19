// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # argmin -- A pure Rust optimization toolbox
//!
//! This crate offers a (work in progress) optimization toolbox/framework written entirely in Rust.
//! It is at the moment probably highly unstable and potentially very buggy. Please use with care
//! and report any bugs you encounter. This crate is looking for contributors!
//!
//! ## Design goals
//!
//! This crate's intention is to be useful to users as well as developers of optimization
//! algorithms, meaning that it should be both easy to apply and easy to implement algoritms. In
//! particular, as a developer of optimization algorithms you should not need to worry about
//! usability features (such as logging, dealing with different types, setters and getters for
//! certain common parameters, counting cost function and gradient evaluations, termination, and so
//! on). Instead you can focus on implementing your algorithm and let argmin do the boring stuff
//! for you.  
//!
//! - Provide an easy framework for the implementation of optimization algorithms: Define a struct
//!   to hold your data, implement a single iteration of your method and let argmin generate the
//!   rest with `#[derive(ArgminSolver)]`. With is approach, the interfaces to different solvers
//!   will be fairly similar, making it easy for users to try different methods on their problem
//!   without much work.
//! - Provide pure Rust implementations of many optimization methods. That way there is no need to
//!   compile and interface C code and it furthermore avoids inconsistent interfaces.
//! - Be type-agnostic: If you have your own special type that you need for solving your
//!   optimization problem, you just need to implement a couple of traits on that type and you're
//!   ready to go. These traits will already be implemented for common types.
//! - Easy iteration information logging: Either print your iteration information to the terminal,
//!   or write it to a file, or store it in a database or send it to a big data pipeline.
//! - Easy evaluation of algorithms: Make it possible to run algorithms with different parameters
//!   and store all necessary of information of all iterations and calculate measures in order to
//!   evaluate the performance of the implementation/method. Take particular care of stochastic
//!   methods.
//!
//! Since this crate is in a very early stage, so far most points are only partially implemented.
//! In addition it is at the moment very likely *very buggy*.

#![warn(missing_docs)]
#![feature(custom_attribute)]
#![feature(unrestricted_attribute_tokens)]
#![allow(unused_attributes)]
// Explicitly disallow EQ comparison of floats. (This clippy lint is denied by default; however,
// this is just to make sure that it will always stay this way.)
#![deny(clippy::float_cmp)]

extern crate argmin_core;
#[macro_use]
extern crate argmin_codegen;
extern crate argmin_testfunctions;
extern crate rand;

/// Definition of all relevant traits
pub mod prelude;

/// Solvers
pub mod solver;

use argmin_core::*;

/// Testfunctions
pub mod testfunctions {
    pub use argmin_testfunctions::*;
}
