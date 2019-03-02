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
//! - `ndarrayl`: Support for `ndarray` and `ndarray-linalg`.
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
//!   gradient at `p`. Optional. By default returns an `Err` if not implemented.
//! - `hessian(&self, p: &Self::Param) -> Result<Self::Hessian, Error>`: Computes the Hessian
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
//! # use serde::{Serialize, Deserialize};
//! // [Imports omited]
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
//!     type Param = ndarray::Array1<f64>;
//!     /// Type of the return value computed by the cost function
//!     type Output = f64;
//!     /// Type of the Hessian. If no Hessian is available or needed for the used solver, this can
//!     /// be set to `()`
//!     type Hessian = ndarray::Array2<f64>;
//!
//!     /// Apply the cost function to a parameter `p`
//!     fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
//!         Ok(rosenbrock_2d(&p.to_vec(), self.a, self.b))
//!     }
//!
//!     /// Compute the gradient at parameter `p`. This is optional: If not implemented, this
//!     /// method will return an `Err` when called.
//!     fn gradient(&self, p: &Self::Param) -> Result<Self::Param, Error> {
//!         Ok(ndarray::Array1::from_vec(rosenbrock_2d_derivative(&p.to_vec(), self.a, self.b)))
//!     }
//!
//!     /// Compute the Hessian at parameter `p`. This is optional: If not implemented, this method
//!     /// will return an `Err` when called.
//!     fn hessian(&self, p: &Self::Param) -> Result<Self::Hessian, Error> {
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
//! use argmin::solver::linesearch::MoreThuenteLineSearch;
//! use serde::{Serialize, Deserialize};
//!
//! #[derive(Clone, Default, Serialize, Deserialize)]
//! struct Rosenbrock {
//!     a: f64,
//!     b: f64,
//! }
//!
//! impl ArgminOp for Rosenbrock {
//!     type Param = ndarray::Array1<f64>;
//!     type Output = f64;
//!     type Hessian = ();
//!
//!    fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
//!        Ok(rosenbrock_2d(&p.to_vec(), self.a, self.b))
//!    }
//!
//!    fn gradient(&self, p: &Self::Param) -> Result<Self::Param, Error> {
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
//!     // Pick a line search.
//!     // let linesearch = HagerZhangLineSearch::new(cost.clone());
//!     let linesearch = MoreThuenteLineSearch::new(cost.clone());
//!     // let linesearch = BacktrackingLineSearch::new(cost.clone());
//!
//!     // Create solver
//!     let mut solver = SteepestDescent::new(cost, init_param, linesearch)?;
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
//! # use argmin::solver::linesearch::MoreThuenteLineSearch;
//! # use serde::{Serialize, Deserialize};
//! #
//! # #[derive(Clone, Default, Serialize, Deserialize)]
//! # struct Rosenbrock {
//! #     a: f64,
//! #     b: f64,
//! # }
//! # impl ArgminOp for Rosenbrock {
//! #     type Param = ndarray::Array1<f64>;
//! #     type Output = f64;
//! #     type Hessian = ();
//! #    fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
//! #        Ok(rosenbrock_2d(&p.to_vec(), self.a, self.b))
//! #    }
//! #    fn gradient(&self, p: &Self::Param) -> Result<Self::Param, Error> {
//! #        Ok(ndarray::Array1::from_vec(rosenbrock_2d_derivative(&p.to_vec(), self.a, self.b)))
//! #    }
//! # }
//! # fn run() -> Result<(), Error> {
//! #     let cost = Rosenbrock { a: 1.0, b: 100.0 };
//! #     let init_param = ndarray::Array1::from_vec(vec![-1.2, 1.0]);
//! #     let linesearch = MoreThuenteLineSearch::new(cost.clone());
//! let mut solver = SteepestDescent::new(cost, init_param, linesearch)?;
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
//! ```
//! // [Imports omited]
//!
//! # extern crate argmin;
//! # extern crate ndarray;
//! # use argmin::prelude::*;
//! # use argmin::solver::linesearch::MoreThuenteLineSearch;
//! # use argmin::solver::quasinewton::BFGS;
//! # use argmin::testfunctions::rosenbrock;
//! # use argmin_core::finitediff::*;
//! # use ndarray::{array, Array1, Array2};
//! # use serde::{Deserialize, Serialize};
//! # #[derive(Clone, Default, Serialize, Deserialize)]
//! # struct Rosenbrock {
//! #     a: f64,
//! #     b: f64,
//! # }
//! # impl ArgminOp for Rosenbrock {
//! #     type Param = Array1<f64>;
//! #     type Output = f64;
//! #     type Hessian = Array2<f64>;
//! #     fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
//! #         Ok(rosenbrock(&p.to_vec(), self.a, self.b))
//! #     }
//! #     fn gradient(&self, p: &Self::Param) -> Result<Self::Param, Error> {
//! #         Ok((*p).forward_diff(&|x| rosenbrock(&x.to_vec(), self.a, self.b)))
//! #     }
//! # }
//! # fn run() -> Result<(), Error> {
//! // Define cost function
//! let cost = Rosenbrock { a: 1.0, b: 100.0 };
//! let init_param: Array1<f64> = array![-1.2, 1.0, -10.0, 2.0, 3.0, 2.0, 4.0, 10.0];
//! let init_hessian: Array2<f64> = Array2::eye(8);
//! let linesearch = MoreThuenteLineSearch::new(cost.clone());
//!
//! // Set up solver
//! let mut solver = BFGS::new(cost, init_param, init_hessian, linesearch);
//!
//! // Set maximum number of iterations
//! solver.set_max_iters(30);
//!
//! // Attach a logger
//! solver.add_logger(ArgminSlogLogger::term());
//!
//! // --------------------------------------------------------------------------------------------
//! // Set up checkpoints
//! // --------------------------------------------------------------------------------------------
//!
//! // Specify the directory where the checkpoints are saved
//! solver.set_checkpoint_dir(".checkpoints");
//!
//! // Specifiy the prefix for each file
//! solver.set_checkpoint_name("bfgs");
//!
//! // Set the `CheckpointMode` which can be `Never` (default),
//! // `Always` (every iteration) or `Every(u64)` (every Nth iteration).
//! solver.set_checkpoint_mode(CheckpointMode::Every(10));
//!
//! // Run solver
//! solver.run()?;
//! # // Wait a second (lets the logger flush everything before printing again)
//! # std::thread::sleep(std::time::Duration::from_secs(1));
//!
//! println!("-------------------------------------------");
//! println!("LOADING CHECKPOINT AND RUNNING SOLVER AGAIN");
//! println!("-------------------------------------------");
//!
//! // now load the same solver from a checkpoint
//! // In order to properly deserialize, the exact type of
//! // the solver needs to be specified.
//! let mut loaded_solver: BFGS<Rosenbrock, MoreThuenteLineSearch<Rosenbrock>> =
//!     BFGS::from_checkpoint(".checkpoints/bfgs.arg")?;
//!
//! // Loggers cannot be serialized, therefore they need to be added again
//! loaded_solver.add_logger(ArgminSlogLogger::term());
//!
//! // Run solver
//! loaded_solver.run()?;
//! # // Wait a second (lets the logger flush everything before printing again)
//! # std::thread::sleep(std::time::Duration::from_secs(1));
//!
//! // Print result
//! println!("-------------------------------------------");
//! println!("Initial run");
//! println!("-------------------------------------------");
//! println!("{}", solver.result());
//!
//! println!("-------------------------------------------");
//! println!("Run from checkpoint");
//! println!("-------------------------------------------");
//! println!("{}", loaded_solver.result());
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
//! A more practical way of using the checkpoints feature is shown in the following example.
//! This will either load an existing checkpoint if one exists or it will create a new solver. Type
//! inference takes care of the return type of `ArgminSolver::from_checkpoint(...)`. This way, the
//! binary can easily be restarted after a crash and will automatically resume from the latest
//! checkpoint.
//!
//! ```rust
//! # extern crate argmin;
//! # extern crate ndarray;
//! # use argmin::prelude::*;
//! # use argmin::solver::linesearch::MoreThuenteLineSearch;
//! # use argmin::solver::quasinewton::BFGS;
//! # use argmin::testfunctions::rosenbrock;
//! # use argmin_core::finitediff::*;
//! # use ndarray::{array, Array1, Array2};
//! # use serde::{Deserialize, Serialize};
//! #
//! # #[derive(Clone, Default, Serialize, Deserialize)]
//! # struct Rosenbrock {
//! #     a: f64,
//! #     b: f64,
//! # }
//! #
//! # impl ArgminOp for Rosenbrock {
//! #     type Param = Array1<f64>;
//! #     type Output = f64;
//! #     type Hessian = Array2<f64>;
//! #
//! #     fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
//! #         Ok(rosenbrock(&p.to_vec(), self.a, self.b))
//! #     }
//! #
//! #     fn gradient(&self, p: &Self::Param) -> Result<Self::Param, Error> {
//! #         Ok((*p).forward_diff(&|x| rosenbrock(&x.to_vec(), self.a, self.b)))
//! #     }
//! # }
//! #
//! # fn run() -> Result<(), Error> {
//! #     // checkpoint directory
//! #
//! #     // Define cost function
//! #     let cost = Rosenbrock { a: 1.0, b: 100.0 };
//! #
//! #     // Define initial parameter vector
//! #     // let init_param: Array1<f64> = array![-1.2, 1.0];
//! #     let init_param: Array1<f64> = array![-1.2, 1.0, -10.0, 2.0, 3.0, 2.0, 4.0, 10.0];
//! #     let init_hessian: Array2<f64> = Array2::eye(8);
//! #
//! #     // set up a line search
//! #     let linesearch = MoreThuenteLineSearch::new(cost.clone());
//! #
//! #     // Set up solver
//! let mut solver = match BFGS::from_checkpoint(".checkpoints/bfgs.arg") {
//!     Ok(solver) => solver,
//!     Err(_) => BFGS::new(cost, init_param, init_hessian, linesearch),
//! };
//! #    Ok(())
//! # }
//! #
//! # fn main() {
//! #     if let Err(ref e) = run() {
//! #         println!("{} {}", e.as_fail(), e.backtrace());
//! #         std::process::exit(1);
//! #     }
//! # }
//!
//! ```
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
//! ```rust
//! // [Imports omited]
//! # extern crate argmin;
//! # extern crate ndarray;
//! # use argmin::prelude::*;
//! # use argmin::solver::linesearch::MoreThuenteLineSearch;
//! # use argmin::solver::quasinewton::BFGS;
//! # use argmin::testfunctions::rosenbrock;
//! # use argmin_core::finitediff::*;
//! # use ndarray::{array, Array1, Array2};
//! # use serde::{Deserialize, Serialize};
//! # use std::sync::Arc;
//! #
//! # #[derive(Clone, Default, Serialize, Deserialize)]
//! # struct Rosenbrock {
//! #     a: f64,
//! #     b: f64,
//! # }
//! #
//! # impl ArgminOp for Rosenbrock {
//! #     type Param = Array1<f64>;
//! #     type Output = f64;
//! #     type Hessian = Array2<f64>;
//! #
//! #     fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
//! #         Ok(rosenbrock(&p.to_vec(), self.a, self.b))
//! #     }
//! #
//! #     fn gradient(&self, p: &Self::Param) -> Result<Self::Param, Error> {
//! #         Ok((*p).forward_diff(&|x| rosenbrock(&x.to_vec(), self.a, self.b)))
//! #     }
//! # }
//! #
//! # fn run() -> Result<(), Error> {
//! #     // Define cost function
//! #     let cost = Rosenbrock { a: 1.0, b: 100.0 };
//! #
//! #     // Define initial parameter vector
//! #     // let init_param: Array1<f64> = array![-1.2, 1.0];
//! #     let init_param: Array1<f64> = array![-1.2, 1.0, -10.0, 2.0, 3.0, 2.0, 4.0, 10.0];
//! #     let init_hessian: Array2<f64> = Array2::eye(8);
//! #
//! #     // set up a line search
//! #     let linesearch = MoreThuenteLineSearch::new(cost.clone());
//! #
//! #     // Set up solver
//! #     let mut solver = BFGS::new(cost, init_param, init_hessian, linesearch);
//! #
//! #     // Set maximum number of iterations
//! #     solver.set_max_iters(10);
//! #
//! #     // Attach a logger
//! #     solver.add_logger(ArgminSlogLogger::term());
//! #
//! // Create writer
//! let mut writer1 = WriteToFile::new("params", "param");
//!
//! // Only save every 3 iterations
//! writer1.set_mode(WriterMode::Every(3));
//!  
//! // Set serializer to JSON
//! writer1.set_serializer(WriteToFileSerializer::JSON);
//!  
//! // Create writer which only saves new best ones
//! let mut writer2 = WriteToFile::new("params", "best");
//!  
//! // Only save new best
//! writer2.set_mode(WriterMode::NewBest);
//!  
//! // Set serializer to `bincode`
//! writer2.set_serializer(WriteToFileSerializer::Bincode);
//!  
//! // Attach writers
//! solver.add_writer(Arc::new(writer1));
//! solver.add_writer(Arc::new(writer2));
//! #
//! #     // Run solver
//! #     solver.run()?;
//! #
//! #     // Wait a second (lets the logger flush everything before printing again)
//! #     std::thread::sleep(std::time::Duration::from_secs(1));
//! #
//! #     // Print result
//! #     println!("{}", solver.result());
//! #     Ok(())
//! # }
//! #
//! # fn main() {
//! #     if let Err(ref e) = run() {
//! #         println!("{} {}", e.as_fail(), e.backtrace());
//! #         std::process::exit(1);
//! #     }
//! # }
//!
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
//! # extern crate argmin_codegen;
//! use argmin_codegen::ArgminSolver;
//! use argmin::prelude::*;
//! use std::default::Default;
//! use serde::{Serialize, Deserialize};
//!
//! // The `Landweber` struct holds the `omega` parameter and has a field `base` which is of type
//! // `ArgminBase`. The struct is generic over the ArgminOp `O` which holds type information about
//! // the parameter vector which (in this particular case) has to implement
//! // `ArgminScaledSub<T, f64>`, which is neede for the update rule.
//! // Deriving `ArgminSolver` implements a large portion of the API and provides many convenience
//! // functions. It requires that `ArgminIter` is implemented on `Landweber` as well.
//! #[derive(ArgminSolver, Serialize, Deserialize)]
//! pub struct Landweber<O>
//! where
//!     O::Param: ArgminScaledSub<O::Param, f64, O::Param>,
//!     O: ArgminOp,
//! {
//!     omega: f64,
//!     base: ArgminBase<O>,
//! }
//!
//! // For convenience, a constructor can/should be implemented
//! impl<O> Landweber<O>
//! where
//!     O::Param: ArgminScaledSub<O::Param, f64, O::Param>,
//!     O: ArgminOp,
//! {
//!     pub fn new(
//!         cost_function: O,
//!         omega: f64,
//!         init_param: O::Param,
//!     ) -> Result<Self, Error> {
//!         Ok(Landweber {
//!             omega,
//!             base: ArgminBase::new(cost_function, init_param),
//!         })
//!     }
//! }
//!
//! // This implements a single iteration of the optimization algorithm.
//! impl<O> ArgminIter for Landweber<O>
//! where
//!     O::Param: ArgminScaledSub<O::Param, f64, O::Param>,
//!     O: ArgminOp,
//! {
//!     type Param = O::Param;
//!     type Output = O::Output;
//!     type Hessian = O::Hessian;
//!
//!     fn next_iter(&mut self) -> Result<ArgminIterData<Self::Param>, Error> {
//!         // Obtain current parameter vector
//!         // The method `cur_param()` has been implemented by deriving `ArgminSolver`.
//!         let param = self.cur_param();
//!         // Compute gradient at current parameter vector `param`
//!         // The method `gradient()` has been implemented by deriving `ArgminSolver`.
//!         let grad = self.gradient(&param)?;
//!         // Calculate new parameter vector based on update rule
//!         let new_param = param.scaled_sub(&self.omega, &grad);
//!         // Return new parameter vector. Since there is no need to compute the cost function
//!         // value, we return 0.0 instead.
//!         let out = ArgminIterData::new(new_param, 0.0);
//!         Ok(out)
//!     }
//! }
//! # fn main() {
//! # }
//! ```

#![warn(missing_docs)]
//#![feature(custom_attribute)]
//#![feature(unrestricted_attribute_tokens)]
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
