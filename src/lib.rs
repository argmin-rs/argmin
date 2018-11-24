// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! A pure Rust optimization framework
//!
//! This crate offers a (work in progress) numerical optimization toolbox/framework written entirely
//! in Rust. It is at the moment quite unstable and potentially very buggy. Please use with care and
//! report any bugs you encounter. This crate is looking for contributors!
//!
//! # Design goals
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
//!   compile and interface C/C++/Fortran code.
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
//! # Algorithms
//!
//! - Linesearches
//!   - Backtracking line search
//!   - More-Thuente line search
//!   - Hager-Zhang line search
//! - Trust region method
//!   - Cauchy point method
//!   - Dogleg method
//!   - Steihaug method
//! - [Steepest Descent](solver/gradientdescent/steepestdescent/struct.SteepestDescent.html)
//! - Conjugate Gradient method
//! - Nonlinear Conjugate Gradient method
//! - Newton Methods
//!   - Basic Newton's Method
//!   - Newton-CG
//! - [Landweber iteration](solver/landweber/struct.Landweber.html)
//! - Simulated Annealing
//!
//! # Usage
//!
//! Add this to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! argmin = "0.1.5"
//! ```
//!
//! ## Optional features
//!
//! There are additional features which can be activated in `Cargo.toml`:
//!
//! ```toml
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
//!
//! # Defining a problem
//!
//! A problem can be defined by implementing the `ArgminOperator` trait which comes with the
//! associated types `Parameters`, `OperatorOutput` and `Hessian`. `Parameters` is the type of your
//! parameter vector (i.e. the input to your cost function), `OperatorOutput` is the type returned
//! by the cost function and `Hessian` is the type of the Hessian.
//! The trait provides the following methods:
//!
//! - `apply(&self, p: &Self::Parameters) -> Result<Self::OperatorOutput, Error>`: Applys the cost
//!   function to parameters `p` of type `Self::Parameters` and returns the cost function value.
//! - `gradient(&self, p: &Self::Parameters) -> Result<Self::Parameters, Error>`: Computes the
//!   gradient at `p`. Optional. By default returns an `Err` if not implemented.
//! - `hessian(&self, p: &Self::Parameters) -> Result<Self::Hessian, Error>`: Computes the Hessian
//!   at `p`. Optional. By default returns an `Err` if not implemented. The type of `Hessian` can
//!   be set to `()` if this method is not implemented.
//!
//!
//! The following code snippet shows an example of how to use the Rosenbrock test functions from
//! `argmin-testfunctions` in argmin:
//!
//! ```rust
//! # extern crate argmin;
//! # extern crate argmin_testfunctions;
//! # extern crate ndarray;
//! # use argmin::testfunctions::{rosenbrock_2d, rosenbrock_2d_derivative, rosenbrock_2d_hessian};
//! # use argmin::prelude::*;
//! // [Imports omited]
//!
//! /// First, create a struct for your problem
//! #[derive(Clone)]
//! struct Rosenbrock {
//!     a: f64,
//!     b: f64,
//! }
//!
//! /// Implement `ArgminOperator` for `Rosenbrock`
//! impl ArgminOperator for Rosenbrock {
//!     /// Type of the parameter vector
//!     type Parameters = ndarray::Array1<f64>;
//!     /// Type of the return value computed by the cost function
//!     type OperatorOutput = f64;
//!     /// Type of the Hessian. If no Hessian is available or needed for the used solver, this can
//!     /// be set to `()`
//!     type Hessian = ndarray::Array2<f64>;
//!
//!     /// Apply the cost function to a parameter `p`
//!     fn apply(&self, p: &Self::Parameters) -> Result<Self::OperatorOutput, Error> {
//!         Ok(rosenbrock_2d(&p.to_vec(), self.a, self.b))
//!     }
//!
//!     /// Compute the gradient at parameter `p`. This is optional: If not implemented, this
//!     /// method will return an `Err` when called.
//!     fn gradient(&self, p: &Self::Parameters) -> Result<Self::Parameters, Error> {
//!         Ok(ndarray::Array1::from_vec(rosenbrock_2d_derivative(&p.to_vec(), self.a, self.b)))
//!     }
//!
//!     /// Compute the Hessian at parameter `p`. This is optional: If not implemented, this method
//!     /// will return an `Err` when called.
//!     fn hessian(&self, p: &Self::Parameters) -> Result<Self::Hessian, Error> {
//!         let h = rosenbrock_2d_hessian(&p.to_vec(), self.a, self.b);
//!         Ok(ndarray::Array::from_shape_vec((2, 2), h).unwrap())
//!     }
//! }
//! ```
//!
//! # Running a solver
//!
//! The following example shows how to use the previously shown definition of a problem in a
//! Steepest Descent (Gradient Descent) solver.
//!
//! ```
//! extern crate argmin;
//! extern crate ndarray;
//! use argmin::testfunctions::{rosenbrock_2d, rosenbrock_2d_derivative, rosenbrock_2d_hessian};
//! use argmin::prelude::*;
//! use argmin::solver::gradientdescent::SteepestDescent;
//!
//! #[derive(Clone)]
//! struct Rosenbrock {
//!     a: f64,
//!     b: f64,
//! }
//!
//! impl ArgminOperator for Rosenbrock {
//!     type Parameters = ndarray::Array1<f64>;
//!     type OperatorOutput = f64;
//!     type Hessian = ();
//!
//!    fn apply(&self, p: &Self::Parameters) -> Result<Self::OperatorOutput, Error> {
//!        Ok(rosenbrock_2d(&p.to_vec(), self.a, self.b))
//!    }
//!
//!    fn gradient(&self, p: &Self::Parameters) -> Result<Self::Parameters, Error> {
//!        Ok(ndarray::Array1::from_vec(rosenbrock_2d_derivative(&p.to_vec(), self.a, self.b)))
//!    }
//! }
//!
//! fn run() -> Result<(), Error> {
//!     // Define cost function
//!     let cost = Rosenbrock { a: 1.0, b: 100.0 };
//!
//!     // Define inital parameter vector
//!     let init_param = ndarray::Array1::from_vec(vec![-1.2, 1.0]);
//!
//!     // Create solver
//!     let mut solver = SteepestDescent::new(&cost, init_param)?;
//!
//!     // Set the maximum number of iterations to 1000
//!     solver.set_max_iters(1000);
//!
//!     // Attach a terminal logger (slog) to the solver
//!     solver.add_logger(ArgminSlogLogger::term());
//!
//!     // Run the solver
//!     solver.run()?;
//!
//!     // Print the result
//!     println!("{:?}", solver.result());
//!     Ok(())
//! }
//!
//! fn main() {
//!     if let Err(ref e) = run() {
//!         println!("{} {}", e.as_fail(), e.backtrace());
//!         std::process::exit(1);
//!     }
//! }
//! ```
//!
//! Executing `solver.run()?` performs the actual optimization. In addition, there is
//! `solver.run_fast()?`, which only executes the optimization algorithm and avoids all convenience
//! functionality such as logging.
//!
//! # Logging
//!
//! Information such as the current iteration number, cost function value, and other metrics can be
//! logged using any object which implements `argmin_core::ArgminLogger`. So far loggers based on
//! the `slog` crate have been implemented: `ArgminSlogLogger::term` logs to the terminal and
//! `ArgminSlogLogger::file` logs to a file in JSON format. Both loggers come with a `*_noblock`
//! version which does not block the execution for logging, but may drop log entries when the
//! buffer fills up.
//!
//! ```
//! # extern crate argmin;
//! # extern crate ndarray;
//! # use argmin::testfunctions::{rosenbrock_2d, rosenbrock_2d_derivative, rosenbrock_2d_hessian};
//! # use argmin::prelude::*;
//! # use argmin::solver::gradientdescent::SteepestDescent;
//! # #[derive(Clone)]
//! # struct Rosenbrock {
//! #     a: f64,
//! #     b: f64,
//! # }
//! # impl ArgminOperator for Rosenbrock {
//! #     type Parameters = ndarray::Array1<f64>;
//! #     type OperatorOutput = f64;
//! #     type Hessian = ();
//! #    fn apply(&self, p: &Self::Parameters) -> Result<Self::OperatorOutput, Error> {
//! #        Ok(rosenbrock_2d(&p.to_vec(), self.a, self.b))
//! #    }
//! #    fn gradient(&self, p: &Self::Parameters) -> Result<Self::Parameters, Error> {
//! #        Ok(ndarray::Array1::from_vec(rosenbrock_2d_derivative(&p.to_vec(), self.a, self.b)))
//! #    }
//! # }
//! # fn run() -> Result<(), Error> {
//! #     let cost = Rosenbrock { a: 1.0, b: 100.0 };
//! #     let init_param = ndarray::Array1::from_vec(vec![-1.2, 1.0]);
//! let mut solver = SteepestDescent::new(&cost, init_param)?;
//! #     solver.set_max_iters(10);
//! // Log to the terminal
//! solver.add_logger(ArgminSlogLogger::term());
//! // Log to the terminal without blocking
//! solver.add_logger(ArgminSlogLogger::term_noblock());
//! // Log to the file `log1.log`
//! solver.add_logger(ArgminSlogLogger::file("log1.log")?);
//! // Log to the file `log2.log` without blocking
//! solver.add_logger(ArgminSlogLogger::file_noblock("log2.log")?);
//! #     solver.run()?;
//! #     Ok(())
//! # }
//! # fn main() {
//! #     if let Err(ref e) = run() {
//! #         println!("{} {}", e.as_fail(), e.backtrace());
//! #         std::process::exit(1);
//! #     }
//! # }
//! ```
//!
//! # Implementing an optimization algorithm
//!
//! In this section we are going to implement the Landweber solver, which essentially is a special
//! form of gradient descent. In iteration `k`, the new parameter vector `x_{k+1}` is calculated
//! from the previous parameter vector `x_k` and the gradient at `x_k` according to the following
//! update rule:
//!
//! `x_{k+1} = x_k - omega * \nabla f(x_k)`
//!
//! In order to implement this using the argmin framework, one first needs to define a struct which
//! holds data/parameters needed during the execution of the algorithm. In addition a field with
//! the name `base` and type `ArgminBase<'a, T, U, H>` is needed, where `T` is the type of the
//! parameter vector, `U` is the type of the return values of the cost function and `H` is the type
//! of the Hessian (which can be `()` if not available).
//!
//! Deriving `ArgminSolver` for the struct using `#[derive(ArgminSolver)]` implements most of the
//! API. What remains to be implemented for the struct is a constructor and `ArgminNextIter`. The
//! latter is essentially an implementation of a single iteration of the algorithm.
//!
//! ```
//! // needed for `#[derive(ArgminSolver)]`
//! #[macro_use]
//! extern crate argmin_codegen;
//! use argmin::prelude::*;
//! use std::default::Default;
//!
//! // The `Landweber` struct holds the `omega` parameter and has a field `base` which is of type
//! // `ArgminBase`. The struct is generic over the paraemter vector (`T`) which (in this
//! // particular case) has to implement `Clone`, `Default` and `ArgminScaledSub<T, f64>`. The
//! // latter is needed for the update rule.
//! // Deriving `ArgminSolver` implements a large portion of the API and provides many convenience
//! // functions. It requires that `ArgminNextIter` is implemented on `Landweber` as well.
//! #[derive(ArgminSolver)]
//! pub struct Landweber<'a, T>
//! where
//!     T: 'a + Clone + Default + ArgminScaledSub<T, f64>,
//! {
//!     omega: f64,
//!     base: ArgminBase<'a, T, f64, ()>,
//! }
//!
//! // For convenience, a constructor can/should be implemented
//! impl<'a, T> Landweber<'a, T>
//! where
//!     T: 'a + Clone + Default + ArgminScaledSub<T, f64>,
//! {
//!     pub fn new(
//!         cost_function: &'a ArgminOperator<Parameters = T, OperatorOutput = f64, Hessian = ()>,
//!         omega: f64,
//!         init_param: T,
//!     ) -> Result<Self, Error> {
//!         Ok(Landweber {
//!             omega,
//!             base: ArgminBase::new(cost_function, init_param),
//!         })
//!     }
//! }
//!
//! // This implements a single iteration of the optimization algorithm.
//! impl<'a, T> ArgminNextIter for Landweber<'a, T>
//! where
//!     T: 'a + Clone + Default + ArgminScaledSub<T, f64>,
//! {
//!     type Parameters = T;
//!     type OperatorOutput = f64;
//!     type Hessian = ();
//!
//!     fn next_iter(&mut self) -> Result<ArgminIterationData<Self::Parameters>, Error> {
//!         // Obtain current parameter vector
//!         // The method `cur_param()` has been implemented by deriving `ArgminSolver`.
//!         let param = self.cur_param();
//!         // Compute gradient at current parameter vector `param`
//!         // The method `gradient()` has been implemented by deriving `ArgminSolver`.
//!         let grad = self.gradient(&param)?;
//!         // Calculate new parameter vector based on update rule
//!         let new_param = param.scaled_sub(self.omega, grad);
//!         // Return new parameter vector. Since there is no need to compute the cost function
//!         // value, we return 0.0 instead.
//!         let out = ArgminIterationData::new(new_param, 0.0);
//!         Ok(out)
//!     }
//! }
//! # fn main() {
//! # }
//! ```

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

/// Definition of all relevant traits and types
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
