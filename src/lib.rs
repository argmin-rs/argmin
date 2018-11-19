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
//! algorithms, meaning that it should be both easy to apply and easy to implement algorithms. In
//! particular, as a developer of optimization algorithms you should not need to worry about
//! usability features (such as logging, dealing with different types, setters and getters for
//! certain common parameters, counting cost function and gradient evaluations, termination, and so
//! on). Instead you can focus on implementing your algorithm and let `argmin-codegen` do the rest.
//!
//! - Easy framework for the implementation of optimization algorithms: Define a struct to hold your
//!   data, implement a single iteration of your method and let argmin generate the rest with
//!   `#[derive(ArgminSolver)]`. This lead to similar interfaces for different solvers, making it
//!   easy for users.
//! - Pure Rust implementations of a wide range of optimization methods: This avoids the need to
//!   compile and interface C code.
//! - Type-agnostic: Many problems require data structures that go beyond simple vectors to
//!   represent the parameters. In argmin, everything is generic: All that needs to be done is
//!   implementing certain traits on your data type. For common types, these traits are already
//!   implemented.
//! - Convenient: Automatic and consistent logging of anything that may be important. Log to the
//!   terminal, to a file or implement your own loggers. Future plans include sending metrics to
//!   databases and connecting to big data piplines.
//! - Algorithm evaluation: Methods to assess the performance of an algorithm for different
//!   parameter settings, problem classes, ...
//!
//! Since this crate is in a very early stage, so far most points are only partially implemented or
//! remain future plans.
//!
//! ## Algorithms
//!
//! - Linesearches
//!   - Backtracking line search
//!   - More-Thuente line search
//!   - Hager-Zhang line search
//! - Trust region method
//!   - Cauchy point method
//!   - Dogleg method
//!   - Steihaug method
//! - Steepest Descent
//! - Conjugate Gradient method
//! - Nonlinear Conjugate Gradient method
//! - Newton Methods
//!   - Basic Newton's Method
//!   - Newton-CG
//! - Landweber iteration
//! - Simulated Annealing
//!
//! ## Usage
//!
//! Add this to your `Cargo.toml`:
//!
//! ```
//! [dependencies]
//! argmin = "0.1.5"
//! ```
//!
//! ### Optional features
//!
//! There are additional features which can be activated in `Cargo.toml`:
//!
//! ```
//! [dependencies]
//! argmin = { version = "0.1.5", features = ["ctrlc", "ndarrayl"] }
//! ```
//!
//! These may become default features in the future. Without these features compilation to
//! `wasm32-unknown-unkown` seems to be possible.
//!
//! - `ctrlc`: Uses the `ctrlc` crate to properly stop the optimization (and return the current best
//!    result) after pressing Ctrl+C.
//! - `ndarrayl`: Support for `ndarray` and `ndarray-linalg`.

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
    //! # Testfunctions
    //!
    //! Reexport of `argmin-testfunctions`.
    pub use argmin_testfunctions::*;
}
