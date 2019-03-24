// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! A pure Rust optimization framework
//!
//! This crate offers a (work in progress) numerical optimization toolbox/framework written entirely
//! in Rust. It is at the moment potentially very buggy. Please use with care and report any bugs
//! you encounter. This crate is looking for contributors!
//!
//! NOTE: The design has changed substantially in the recent past and the documentation is not yet
//! up to date. If you are using a version available on
//! [crates.io](https://crates.io/crates/argmin) please have a look at the correspoding
//! [documentation](https://docs.rs/argmin/0.1.8/argmin/).
//!
//! # Design goals
//!
//! This crate's intention is to be useful to users as well as developers of optimization
//! algorithms, meaning that it should be both easy to apply and easy to implement algorithms. In
//! particular, as a developer of optimization algorithms you should not need to worry about
//! usability features (such as logging, dealing with different types, setters and getters for
//! certain common parameters, counting cost function and gradient evaluations, termination, and so
//! on). Instead you can focus on implementing your algorithm.
//!
//! - Easy framework for the implementation of optimization algorithms: Implement a single iteration
//!   of your method and let the framework do the rest. This leads to similar interfaces for
//!   different solvers, making it easy for users.
//! - Pure Rust implementations of a wide range of optimization methods: This avoids the need to
//!   compile and interface C/C++/Fortran code.
//! - Type-agnostic: Many problems require data structures that go beyond simple vectors to
//!   represent the parameters. In argmin, everything is generic: All that needs to be done is
//!   implementing certain traits on your data type. For common types, these traits are already
//!   implemented.
//! - Convenient: Easy and consistent logging of anything that may be important. Log to the
//!   terminal, to a file or implement your own observers. Future plans include sending metrics to
//!   databases and connecting to big data piplines.
//! - Algorithm evaluation: Methods to assess the performance of an algorithm for different
//!   parameter settings, problem classes, ...
//!
//! Since this crate is in a very early stage, so far most points are only partially implemented or
//! remain future plans.
//!
//! # Algorithms
//!
//! - [Line searches](solver/linesearch/index.html)
//!   - [Backtracking line search](solver/linesearch/backtracking/struct.BacktrackingLineSearch.html)
//!   - [More-Thuente line search](solver/linesearch/morethuente/struct.MoreThuenteLineSearch.html)
//!   - [Hager-Zhang line search](solver/linesearch/hagerzhang/struct.HagerZhangLineSearch.html)
//! - [Trust region method](solver/trustregion/trustregion_method/struct.TrustRegion.html)
//!   - [Cauchy point method](solver/trustregion/cauchypoint/struct.CauchyPoint.html)
//!   - [Dogleg method](solver/trustregion/dogleg/struct.Dogleg.html)
//!   - [Steihaug method](solver/trustregion/steihaug/struct.Steihaug.html)
//! - [Steepest descent](solver/gradientdescent/steepestdescent/struct.SteepestDescent.html)
//! - [Conjugate gradient method](solver/conjugategradient/cg/struct.ConjugateGradient.html)
//! - [Nonlinear conjugate gradient method](solver/conjugategradient/nonlinear_cg/struct.NonlinearConjugateGradient.html)
//! - [Newton methods](solver/newton/index.html)
//!   - [Newton's method](solver/newton/newton_method/struct.Newton.html)
//!   - [Newton-CG](solver/newton/newton_cg/struct.NewtonCG.html)
//! - [Quasi-Newton methods](solver/quasinewton/index.html)
//!   - [BFGS](solver/quasinewton/bfgs/struct.BFGS.html)
//!   - [DFP](solver/quasinewton/dfp/struct.DFP.html)
//! - [Landweber iteration](solver/landweber/struct.Landweber.html)
//! - [Simulated Annealing](solver/simulatedannealing/struct.SimulatedAnnealing.html)
//!
//! # Usage
//!
//! Add this to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! argmin = "0.1.8"
//! ```
//!
//! ## Optional features
//!
//! There are additional features which can be activated in `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! argmin = { version = "0.1.8", features = ["ctrlc", "ndarrayl"] }
//! ```
//!
//! These may become default features in the future. Without these features compilation to
//! `wasm32-unknown-unkown` seems to be possible.
//!
//! - `ctrlc`: Uses the `ctrlc` crate to properly stop the optimization (and return the current best
//!    result) after pressing Ctrl+C.
//! - `ndarrayl`: Support for `ndarray`, `ndarray-linalg` and `ndarray-rand`.
//!
//! # Defining a problem
//!
//! A problem can be defined by implementing the `ArgminOp` trait which comes with the
//! associated types `Param`, `Output` and `Hessian`. `Param` is the type of your
//! parameter vector (i.e. the input to your cost function), `Output` is the type returned
//! by the cost function and `Hessian` is the type of the Hessian.
//! The trait provides the following methods:
//!
//! - `apply(&self, p: &Self::Param) -> Result<Self::Output, Error>`: Applys the cost
//!   function to parameters `p` of type `Self::Param` and returns the cost function value.
//! - `gradient(&self, p: &Self::Param) -> Result<Self::Param, Error>`: Computes the
//!   gradient at `p`.
//! - `hessian(&self, p: &Self::Param) -> Result<Self::Hessian, Error>`: Computes the Hessian
//!   at `p`.
//!
//! The following code snippet shows an example of how to use the Rosenbrock test functions from
//! `argmin-testfunctions` in argmin:
//!
//! ```rust
//! # extern crate argmin;
//! # extern crate argmin_testfunctions;
//! # extern crate ndarray;
//! use argmin::testfunctions::{rosenbrock_2d, rosenbrock_2d_derivative, rosenbrock_2d_hessian};
//! use argmin::prelude::*;
//! use serde::{Serialize, Deserialize};
//!
//! /// First, create a struct for your problem
//! #[derive(Clone, Default, Serialize, Deserialize)]
//! struct Rosenbrock {
//!     a: f64,
//!     b: f64,
//! }
//!
//! /// Implement `ArgminOp` for `Rosenbrock`
//! impl ArgminOp for Rosenbrock {
//!     /// Type of the parameter vector
//!     type Param = Vec<f64>;
//!     /// Type of the return value computed by the cost function
//!     type Output = f64;
//!     /// Type of the Hessian. Can be `()` if not needed.
//!     type Hessian = Vec<Vec<f64>>;
//!
//!     /// Apply the cost function to a parameter `p`
//!     fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
//!         Ok(rosenbrock_2d(p, self.a, self.b))
//!     }
//!
//!     /// Compute the gradient at parameter `p`.
//!     fn gradient(&self, p: &Self::Param) -> Result<Self::Param, Error> {
//!         Ok(rosenbrock_2d_derivative(p, self.a, self.b))
//!     }
//!
//!     /// Compute the Hessian at parameter `p`.
//!     fn hessian(&self, p: &Self::Param) -> Result<Self::Hessian, Error> {
//!         let t = rosenbrock_2d_hessian(p, self.a, self.b);
//!         Ok(vec![vec![t[0], t[1]], vec![t[2], t[3]]])
//!     }
//! }
//! ```
//!
//! It is optional to implement any of these methods, as there are default implementations which
//! will return an `Err` when called. What needs to be implemented is defined by the requirements
//! of the solver that is to be used.
//!
//! # Running a solver
//!
//! The following example shows how to use the previously shown definition of a problem in a
//! Steepest Descent (Gradient Descent) solver.
//!
//! ```rust
//! # #![allow(unused_imports)]
//! # extern crate argmin;
//! use argmin::prelude::*;
//! use argmin::solver::gradientdescent::SteepestDescent;
//! use argmin::solver::linesearch::MoreThuenteLineSearch;
//! # use argmin::testfunctions::{rosenbrock_2d, rosenbrock_2d_derivative};
//! # use serde::{Deserialize, Serialize};
//! #
//! # #[derive(Clone, Default, Serialize, Deserialize)]
//! # struct Rosenbrock {
//! #     a: f64,
//! #     b: f64,
//! # }
//! #
//! # impl ArgminOp for Rosenbrock {
//! #     type Param = Vec<f64>;
//! #     type Output = f64;
//! #     type Hessian = ();
//! #
//! #     fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
//! #         Ok(rosenbrock_2d(p, self.a, self.b))
//! #     }
//! #
//! #     fn gradient(&self, p: &Self::Param) -> Result<Self::Param, Error> {
//! #         Ok(rosenbrock_2d_derivative(p, self.a, self.b))
//! #     }
//! # }
//! #
//! # fn run() -> Result<(), Error> {
//!
//! // Define cost function (must implement `ArgminOperator`)
//! let cost = Rosenbrock { a: 1.0, b: 100.0 };
//!  
//! // Define initial parameter vector
//! let init_param: Vec<f64> = vec![-1.2, 1.0];
//!  
//! // Set up line search
//! let linesearch = MoreThuenteLineSearch::new();
//!  
//! // Set up solver
//! let solver = SteepestDescent::new(linesearch);
//!  
//! // Run solver
//! let res = Executor::new(cost, solver, init_param)
//!     // Add an observer which will log all iterations to the terminal
//!     .add_observer(ArgminSlogLogger::term(), ObserverMode::Always)
//!     // Set maximum iterations to 10
//!     .max_iters(10)
//!     // run the solver on the defined problem
//!     .run()?;
//! #
//! #     // Wait a second (lets the logger flush everything first)
//! #     std::thread::sleep(std::time::Duration::from_secs(1));
//!  
//! // print result
//! println!("{}", res);
//! #     Ok(())
//! # }
//! #
//! # fn main() {
//! #     if let Err(ref e) = run() {
//! #         println!("{} {}", e.as_fail(), e.backtrace());
//! #         std::process::exit(1);
//! #     }
//! # }
//! ```
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
// //! ```
// //! unimplemented!()
// //! ```
//!
//! # Checkpoints
//!
//! The longer an optimization runs, the higher the probability that something crashes.
//! Particularly for optimizations which are running for days, weeks or even longer, this can
//! become a problem. To mitigate this problem, it is possible in argmin to save checkpoints.
//! Such a checkpoint is a serialization of an `ArgminSolver` object and can be loaded again and
//! resumed.
//! The `CheckpointMode` defines how often checkpoints are saved and is either `Never` (default),
//! `Always` (every iteration) or `Every(u64)` (every Nth iteration). It is set via the setter
//! method `set_checkpoint_mode()` which is implemented for every `ArgminSolver`.
//! In addition, the directory where the checkpoints and a prefix for every file can be set via
//! `set_checkpoint_dir()` and `set_checkpoint_prefix`, respectively.
//!
//! The following example illustrates the usage. Note that this example is only for illustration
//! and does not make much sense. Please scroll down for a more practical example.
//!
// //! ```
// //! // [Imports omited]
// //! TODO
// //! ```
//!
//! A more practical way of using the checkpoints feature is shown in the following example.
//! This will either load an existing checkpoint if one exists or it will create a new solver. Type
//! inference takes care of the return type of `ArgminSolver::from_checkpoint(...)`. This way, the
//! binary can easily be restarted after a crash and will automatically resume from the latest
//! checkpoint.
//!
// //! ```rust
// //! unimplemented!()
// //! ```
//!
//! # Writers
//!
//! Writers can be used to handle parameter vectors in some way during the optimization
//! (suggestions for a better name are more than welcome!). Usually, this can be used to save the
//! intermediate parameter vectors somewhere. Currently, different modes are supported:
//!
//! * `WriterMode::Never`: Don't do anything.
//! * `WriterMode::Always`: Process parameter vector in every iteration.
//! * `WriterMode::Every(i)`: Process parameter vector in every i-th iteration.
//! * `WriterMode::NewBest`: Process parameter vector whenever there is a new best one.
//!
//! The following example creates two writers of the type `WriteToFile` which serializes the
//! parameter vector using either `serde_json` or `bincode`. The first writer saves the parameters
//! in every third iteration (as JSON), while the second one saves only the new best ones (using
//! `bincode`).
//! Both are attached to a solver using the `add_writer(...)` method of `ArgminSolver` before the
//! solver is run.
//!
// //! ```rust
// //! // [Imports omited]
// //! TODO
// //! ```
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
// //! ```rust
// //! // [Imports omited]
// //! TODO
// //! ```

#![warn(missing_docs)]
#![allow(unused_attributes)]
// Explicitly disallow EQ comparison of floats. (This clippy lint is denied by default; however,
// this is just to make sure that it will always stay this way.)
#![deny(clippy::float_cmp)]

extern crate argmin_core;
extern crate argmin_testfunctions;
extern crate rand;

/// Definition of all relevant traits and types
pub mod prelude;

/// Solvers
pub mod solver;

/// Macros
#[macro_use]
mod macros;

use argmin_core::*;

/// Testfunctions
pub mod testfunctions {
    //! # Testfunctions
    //!
    //! Reexport of `argmin-testfunctions`.
    pub use argmin_testfunctions::*;
}
