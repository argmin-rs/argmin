// Copyright 2018-2022 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! argmin is a numerical optimization library written entirely in Rust.
//!
//! [Documentation of most recent release](https://docs.rs/argmin/latest/argmin/)
//!
//! [Documentation of main branch](https://argmin-rs.github.io/argmin/argmin/)
//!
//! # Design goals
//!
//! argmin aims at offering a wide range of optimization algorithms with a consistent interface,
//! written purely in Rust. It comes with additional features such as checkpointing and observers
//! which for instance make it possible to log the progress of an optimization to screen or file.
//!
//! It further provides a framework for implementing iterative optimization algorithms in a
//! convenient manner. Essentially, a single iteration of the algorithm needs to be implemented and
//! everything else, such as handling termination, parameter vectors, gradients and Hessians, is
//! taken care of by the library.
//!
//! This library uses generics to be as type-agnostic as possible. Abstractions over common math
//! functions enable the use of common backends such as `ndarray` and `nalgebra` via the
//! `argmin-math` crate. All operations can be performed with 32 and 64 bit floats. Custom types are
//! of course also supported.
//!
//! # Contributing
//!
//! This crate is looking for contributors!
//! Potential projects can be found in the
//! [Github issues](https://github.com/argmin-rs/argmin/issues), but feel free to suggest your own
//! ideas as well. Besides adding optimization methods and new features, other contributions are
//! also highly welcome, for instance improving performance, documentation, writing examples (with
//! real world problems), developing tests, adding observers, implementing a C interface or
//! [Python wrappers](https://github.com/argmin-rs/pyargmin).
//! Bug reports (and fixes) are of course also highly appreciated.
//!
//! # Algorithms
//!
//! - [Line searches](solver/linesearch/index.html)
//!
//!   - [Backtracking line search](solver/linesearch/backtracking/struct.BacktrackingLineSearch.html)
//!   - [More-Thuente line search](solver/linesearch/morethuente/struct.MoreThuenteLineSearch.html)
//!   - [Hager-Zhang line search](solver/linesearch/hagerzhang/struct.HagerZhangLineSearch.html)
//!
//! - [Trust region method](solver/trustregion/trustregion_method/struct.TrustRegion.html)
//!
//!   - [Cauchy point method](solver/trustregion/cauchypoint/struct.CauchyPoint.html)
//!   - [Dogleg method](solver/trustregion/dogleg/struct.Dogleg.html)
//!   - [Steihaug method](solver/trustregion/steihaug/struct.Steihaug.html)
//!   
//! - [Steepest descent](solver/gradientdescent/steepestdescent/struct.SteepestDescent.html)
//!
//! - [Conjugate gradient method](solver/conjugategradient/cg/struct.ConjugateGradient.html)
//!
//! - [Nonlinear conjugate gradient method](solver/conjugategradient/nonlinear_cg/struct.NonlinearConjugateGradient.html)
//!
//! - [Newton methods](solver/newton/index.html)
//!
//!   - [Newton's method](solver/newton/newton_method/struct.Newton.html)
//!   - [Newton-CG](solver/newton/newton_cg/struct.NewtonCG.html)
//!
//! - [Quasi-Newton methods](solver/quasinewton/index.html)
//!
//!   - [BFGS](solver/quasinewton/bfgs/struct.BFGS.html)
//!   - [L-BFGS](solver/quasinewton/lbfgs/struct.LBFGS.html)
//!   - [DFP](solver/quasinewton/dfp/struct.DFP.html)
//!   - [SR1](solver/quasinewton/sr1/struct.SR1.html)
//!   - [SR1-TrustRegion](solver/quasinewton/sr1_trustregion/struct.SR1TrustRegion.html)
//!
//! - [Gauss-Newton method](solver/gaussnewton/gaussnewton_method/struct.GaussNewton.html)
//!
//! - [Gauss-Newton method with linesearch](solver/gaussnewton/gaussnewton_linesearch/struct.GaussNewtonLS.html)
//!
//! - [Golden-section search](solver/goldensectionsearch/struct.GoldenSectionSearch.html)
//!
//! - [Landweber iteration](solver/landweber/struct.Landweber.html)
//!
//! - [Brent's method](solver/brent/struct.Brent.html)
//!
//! - [Nelder-Mead method](solver/neldermead/struct.NelderMead.html)
//!
//! - [Simulated Annealing](solver/simulatedannealing/struct.SimulatedAnnealing.html)
//!
//! - [Particle Swarm Optimization](solver/particleswarm/struct.ParticleSwarm.html)
//!
//! # Examples
//!
//! Examples for each solver can be found
//! [here (current released version)](https://github.com/argmin-rs/argmin/tree/v0.5.0/examples) and
//! [here (main branch)](https://github.com/argmin-rs/argmin/tree/main/argmin/examples).
//!
//! # Usage
//!
//! Add this to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
#![doc = concat!(" argmin = \"", env!("CARGO_PKG_VERSION"), "\"")]
//! argmin-math = { version = "0.1.0", features = ["ndarray_latest-serde,nalgebra_latest-serde"] }
//! ```
//!
//! or, for the current development version:
//!
//! ```toml
//! [dependencies]
//! argmin = { git = "https://github.com/argmin-rs/argmin" }
//! argmin-math = { git = "https://github.com/argmin-rs/argmin", features = ["ndarray_latest-serde,nalgebra_latest-serde"] }
//! ```
//!
//! (For which features to select for `argmin-math` please see the
//! [documentation](https://docs.rs/argmin/latest/argmin-math).)
//!
//! ## Features
//!
//! ### Default features
//!
//! - `slog-logger`: Support for logging using `slog`
//! - `serde1`: Support for `serde`. Needed for checkpointing and writing parameters to disk as
//! well as logging to disk.
//!
//! ### Optional features
//!
//! The `ctrlc` feature uses the `ctrlc` crate to properly stop the optimization (and return the
//! current best result) after pressing Ctrl+C during an optimization run.
//!
//! ```toml
//! [dependencies]
#![doc = concat!(" argmin = { version = \"", env!("CARGO_PKG_VERSION"), "\", features = [\"ctrlc\"] }")]
//! ```
//!
//! ### Experimental support for compiling to WebAssembly
//!
//! When compiling to WASM, the feature `wasm-bindgen` must be used.
//!
//! WASM support is still experimental. Please report any issues you encounter when using argmin
//! in a WASM context.
//!
//! ### Compiling without `serde` dependency
//!
//! The `serde` dependency can be removed by turning off the `serde1` feature, for instance like so:
//!
//! ```toml
//! [dependencies]
#![doc = concat!(" argmin = { version = \"", env!("CARGO_PKG_VERSION"), "\", default-features = false, features = [\"slog-logger\"] }")]
//! ```
//!
//! Note that this will remove the ability to write parameters and logs to disk as well as
//! checkpointing.
//!
//! ## Running the tests and building the examples
//!
//! The tests and examples require a set of features to be enabled:
//!
//! ```bash
//! cargo test --features "argmin/ctrlc,argmin-math/ndarray_latest-serde,argmin-math/nalgebra_latest-serde,argmin/ndarrayl"
//! ```
//!
//! # Defining a problem
//!
//! TODO TODO TODO TODO Rewrite!
//! A problem can be defined by implementing the `ArgminOp` trait which comes with the
//! associated types `Param`, `Output` and `Hessian`. `Param` is the type of your
//! parameter vector (i.e. the input to your cost function), `Output` is the type returned
//! by the cost function, `Hessian` is the type of the Hessian and `Jacobian` is the type of the
//! Jacobian.
//! The trait provides the following methods:
//!
//! - `apply(&self, p: &Self::Param) -> Result<Self::Output, Error>`: Applys the cost
//!   function to parameters `p` of type `Self::Param` and returns the cost function value.
//! - `gradient(&self, p: &Self::Param) -> Result<Self::Param, Error>`: Computes the
//!   gradient at `p`.
//! - `hessian(&self, p: &Self::Param) -> Result<Self::Hessian, Error>`: Computes the Hessian
//!   at `p`.
//! - `jacobian(&self, p: &Self::Param) -> Result<Self::Jacobian, Error>`: Computes the Jacobian
//!   at `p`.
//!
//! The following code snippet shows an example of how to use the Rosenbrock test functions from
//! `argmin-testfunctions` in argmin:
//!
//! ```rust
//! # extern crate argmin;
//! # extern crate argmin_testfunctions;
//! use argmin_testfunctions::{rosenbrock_2d, rosenbrock_2d_derivative, rosenbrock_2d_hessian};
//! use argmin::core::{Error, CostFunction, Gradient, Hessian};
//!
//! /// First, create a struct for your problem
//! struct Rosenbrock {
//!     a: f64,
//!     b: f64,
//! }
//!
//! /// Implement `CostFunction` for `Rosenbrock`
//! impl CostFunction for Rosenbrock {
//!     /// Type of the parameter vector
//!     type Param = Vec<f64>;
//!     /// Type of the return value computed by the cost function
//!     type Output = f64;
//!
//!     /// Apply the cost function to a parameter `p`
//!     fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
//!         Ok(rosenbrock_2d(p, 1.0, 100.0))
//!     }
//! }
//!
//! /// Implement `Gradient` for `Rosenbrock`
//! impl Gradient for Rosenbrock {
//!     /// Type of the parameter vector
//!     type Param = Vec<f64>;
//!     /// Type of the return value computed by the cost function
//!     type Gradient = Vec<f64>;
//!
//!     /// Compute the gradient at parameter `p`.
//!     fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
//!         Ok(rosenbrock_2d_derivative(p, 1.0, 100.0))
//!     }
//! }
//!
//! /// Implement `Hessian` for `Rosenbrock`
//! impl Hessian for Rosenbrock {
//!     /// Type of the parameter vector
//!     type Param = Vec<f64>;
//!     /// Type of the Hessian. Can be `()` if not needed.
//!     type Hessian = Vec<Vec<f64>>;
//!
//!     /// Compute the Hessian at parameter `p`.
//!     fn hessian(&self, p: &Self::Param) -> Result<Self::Hessian, Error> {
//!         let t = rosenbrock_2d_hessian(p, 1.0, 100.0);
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
//! # extern crate argmin_testfunctions;
//! use argmin::core::{Error, Executor, CostFunction, Gradient};
//! # #[cfg(feature = "slog-logger")]
//! use argmin::core::{ArgminSlogLogger, ObserverMode};
//! use argmin::solver::gradientdescent::SteepestDescent;
//! use argmin::solver::linesearch::MoreThuenteLineSearch;
//! # use argmin_testfunctions::{rosenbrock_2d, rosenbrock_2d_derivative};
//! #
//! # struct Rosenbrock {
//! #     a: f64,
//! #     b: f64,
//! # }
//! #
//! # /// Implement `CostFunction` for `Rosenbrock`
//! # impl CostFunction for Rosenbrock {
//! #     /// Type of the parameter vector
//! #     type Param = Vec<f64>;
//! #     /// Type of the return value computed by the cost function
//! #     type Output = f64;
//! #
//! #     /// Apply the cost function to a parameter `p`
//! #     fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
//! #         Ok(rosenbrock_2d(p, 1.0, 100.0))
//! #     }
//! # }
//! #
//! # /// Implement `Gradient` for `Rosenbrock`
//! # impl Gradient for Rosenbrock {
//! #     /// Type of the parameter vector
//! #     type Param = Vec<f64>;
//! #     /// Type of the return value computed by the cost function
//! #     type Gradient = Vec<f64>;
//! #
//! #     /// Compute the gradient at parameter `p`.
//! #     fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
//! #         Ok(rosenbrock_2d_derivative(p, 1.0, 100.0))
//! #     }
//! # }
//! #
//! # fn run() -> Result<(), Error> {
//!
//! // Define cost function (must implement `CostFunction` and `Gradient`)
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
//! let res = Executor::new(cost, solver)
//!     .configure(|config|
//!         config
//!             // Set initial parameters
//!             .param(init_param)
//!             // Set maximum iterations to 10
//!             .max_iters(10)
//!     )
//! # ;
//! # #[cfg(feature = "slog-logger")]
//! # let res = res
//!     // Add an observer which will log all iterations to the terminal
//!     .add_observer(ArgminSlogLogger::term(), ObserverMode::Always)
//! # ;
//! # let res = res
//!     // run the solver on the defined problem
//!     .run()?;
//! #
//! // print result
//! println!("{}", res);
//! #     Ok(())
//! # }
//! #
//! # fn main() {
//! #     if let Err(ref e) = run() {
//! #         println!("{}", e);
//! #         std::process::exit(1);
//! #     }
//! # }
//! ```
//!
//! # Observing iterations
//!
//! Argmin offers an interface to observe the state of the solver at initialization as well as
//! after every iteration. This includes the parameter vector, gradient, Hessian, iteration number,
//! cost values and many more as well as solver-specific metrics. This interface can be used to
//! implement loggers, send the information to a storage or to plot metrics.
//! Observers need to implement the `Observe` trait.
//! Argmin ships with a logger based on the `slog` crate. `ArgminSlogLogger::term` logs to the
//! terminal and `ArgminSlogLogger::file` logs to a file in JSON format. Both loggers also come
//! with a `*_noblock` version which does not block the execution of logging, but may drop some
//! messages in case of a full buffer.
//! Parameter vectors can be written to disk using `WriteToFile`.
//! For each observer it can be defined how often it will observe the progress of the solver. This
//! is indicated via the enum `ObserverMode` which can be either `Always`, `Never`, `NewBest`
//! (whenever a new best solution is found) or `Every(i)` which means every `i`th iteration.
//!
//! ```rust
//! # #![allow(unused_imports)]
//! # extern crate argmin;
//! # extern crate argmin_testfunctions;
//! # use argmin::core::{Error, Executor, CostFunction, Gradient, ObserverMode};
//! # #[cfg(feature = "slog-logger")]
//! # use argmin::core::ArgminSlogLogger;
//! # #[cfg(feature = "serde1")]
//! # use argmin::core::WriteToFile;
//! # use argmin::solver::gradientdescent::SteepestDescent;
//! # use argmin::solver::linesearch::MoreThuenteLineSearch;
//! # use argmin_testfunctions::{rosenbrock_2d, rosenbrock_2d_derivative};
//! #
//! # struct Rosenbrock {
//! #     a: f64,
//! #     b: f64,
//! # }
//! #
//! # /// Implement `CostFunction` for `Rosenbrock`
//! # impl CostFunction for Rosenbrock {
//! #     /// Type of the parameter vector
//! #     type Param = Vec<f64>;
//! #     /// Type of the return value computed by the cost function
//! #     type Output = f64;
//! #
//! #     /// Apply the cost function to a parameter `p`
//! #     fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
//! #         Ok(rosenbrock_2d(p, 1.0, 100.0))
//! #     }
//! # }
//! #
//! # /// Implement `Gradient` for `Rosenbrock`
//! # impl Gradient for Rosenbrock {
//! #     /// Type of the parameter vector
//! #     type Param = Vec<f64>;
//! #     /// Type of the return value computed by the cost function
//! #     type Gradient = Vec<f64>;
//! #
//! #     /// Compute the gradient at parameter `p`.
//! #     fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
//! #         Ok(rosenbrock_2d_derivative(p, 1.0, 100.0))
//! #     }
//! # }
//! #
//! # fn run() -> Result<(), Error> {
//! #
//! # // Define cost function (must implement `CostFunction` and `Gradient`)
//! # let problem = Rosenbrock { a: 1.0, b: 100.0 };
//! #
//! # // Define initial parameter vector
//! # let init_param: Vec<f64> = vec![-1.2, 1.0];
//! #
//! # // Set up line search
//! # let linesearch = MoreThuenteLineSearch::new();
//! #
//! # // Set up solver
//! # let solver = SteepestDescent::new(linesearch);
//! #
//! let res = Executor::new(problem, solver)
//!     .configure(|config| config.param(init_param).max_iters(2))
//! # ;
//! # #[cfg(feature = "slog-logger")]
//! # let res = res
//!     // Add an observer which will log all iterations to the terminal (without blocking)
//!     .add_observer(ArgminSlogLogger::term_noblock(), ObserverMode::Always)
//! # ;
//! # #[cfg(feature = "serde1")]
//! # #[cfg(feature = "slog-logger")]
//! # let res = res
//!     // Log to file whenever a new best solution is found
//!     .add_observer(ArgminSlogLogger::file("solver.log", false)?, ObserverMode::NewBest)
//!     // Write parameter vector to `params/param.arg` every 20th iteration
//!     .add_observer(WriteToFile::new("params", "param"), ObserverMode::Every(20))
//! # ;
//! # let res = res
//!     // run the solver on the defined problem
//!     .run()?;
//! #     Ok(())
//! # }
//! #
//! # fn main() {
//! #     if let Err(ref e) = run() {
//! #         println!("{}", e);
//! #         std::process::exit(1);
//! #     }
//! # }
//! ```
//!
//! # Checkpoints
//!
//! The probability of crashes increases with runtime, therefore one may want to save checkpoints
//! in order to be able to resume the optimization after a crash.
//! The `CheckpointMode` defines how often checkpoints are saved and is either `Never` (default),
//! `Always` (every iteration) or `Every(u64)` (every Nth iteration). It is set via the setter
//! method `checkpoint_mode` of `Executor`.
//! In addition, the directory where the checkpoints and a prefix for every file can be set via
//! `checkpoint_dir` and `checkpoint_name`, respectively.
//!
//! The following example shows how the `from_checkpoint` method can be used to resume from a
//! checkpoint. In case this fails (for instance because the file does not exist, which could mean
//! that this is the first run and there is nothing to resume from), it will resort to creating a
//! new `Executor`, thus starting from scratch.
//!
//! ```rust
//! # extern crate argmin;
//! # extern crate argmin_testfunctions;
//! # use argmin::core::{CostFunction, Error, Executor, Gradient, ObserverMode};
//! # #[cfg(feature = "serde1")]
//! # use argmin::core::{CheckpointMode};
//! # #[cfg(feature = "slog-logger")]
//! # use argmin::core::{ArgminSlogLogger};
//! # use argmin::solver::landweber::Landweber;
//! # use argmin_testfunctions::{rosenbrock_2d, rosenbrock_2d_derivative};
//! #
//! # #[derive(Default)]
//! # struct Rosenbrock {}
//! #
//! # /// Implement `CostFunction` for `Rosenbrock`
//! # impl CostFunction for Rosenbrock {
//! #     /// Type of the parameter vector
//! #     type Param = Vec<f64>;
//! #     /// Type of the return value computed by the cost function
//! #     type Output = f64;
//! #
//! #     /// Apply the cost function to a parameter `p`
//! #     fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
//! #         Ok(rosenbrock_2d(p, 1.0, 100.0))
//! #     }
//! # }
//! #
//! # /// Implement `Gradient` for `Rosenbrock`
//! # impl Gradient for Rosenbrock {
//! #     /// Type of the parameter vector
//! #     type Param = Vec<f64>;
//! #     /// Type of the return value computed by the cost function
//! #     type Gradient = Vec<f64>;
//! #
//! #     /// Compute the gradient at parameter `p`.
//! #     fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
//! #         Ok(rosenbrock_2d_derivative(p, 1.0, 100.0))
//! #     }
//! # }
//! #
//! # fn run() -> Result<(), Error> {
//! #     // define inital parameter vector
//! #     let init_param: Vec<f64> = vec![1.2, 1.2];
//! #
//! #     let iters = 35;
//! #     let solver = Landweber::new(0.001);
//! #
//! # #[cfg(feature = "serde1")]
//! let res = Executor::from_checkpoint(".checkpoints/optim.arg", Rosenbrock {})
//!     .unwrap_or(
//!         Executor::new(Rosenbrock {}, solver).configure(
//!             |config| config.param(init_param).max_iters(iters)
//!         )
//!     )
//!     .checkpoint_dir(".checkpoints")
//!     .checkpoint_name("optim")
//!     .checkpoint_mode(CheckpointMode::Every(20))
//!     .run()?;
//! #
//! #     Ok(())
//! # }
//! #
//! # fn main() {
//! #     if let Err(ref e) = run() {
//! #         println!("{}", e);
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
//! holds data specific to the solver. Then, the `Solver` trait needs to be implemented for the
//! struct. This requires setting the associated constant `NAME` which gives your solver a name.
//! The `next_iter` method defines the computations performed in a single iteration of the solver.
//! Via the parameters `op` and `state` one has access to the operator (cost function, gradient
//! computation, Hessian, ...) and to the current state of the optimization (parameter vectors,
//! cost function values, iteration number, ...), respectively.
//!
//! ```rust
//! use argmin::core::{
//!     ArgminFloat, ArgminKV, Error, Gradient, IterState, OpWrapper, Solver, State
//! };
//! #[cfg(feature = "serde1")]
//! use serde::{Deserialize, Serialize};
//! use argmin_math::ArgminScaledSub;
//!
//! // Define a struct which holds any parameters/data which are needed during the execution of the
//! // solver. Note that this does not include parameter vectors, gradients, Hessians, cost
//! // function values and so on, as those will be handled by the `Executor`.
//! #[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
//! pub struct Landweber<F> {
//!     /// omega
//!     omega: F,
//! }
//!
//! impl<F> Landweber<F> {
//!     /// Constructor
//!     pub fn new(omega: F) -> Self {
//!         Landweber { omega }
//!     }
//! }
//!
//! impl<O, F, P, G> Solver<O, IterState<P, G, (), (), F>> for Landweber<F>
//! where
//!     // `O` needs to implement `Gradient` for the Landweber solver
//!     O: Gradient<Param = P, Gradient = G>,
//!     // `P` needs to implement `ArgminScaledSub` because of the update formula
//!     P: Clone + ArgminScaledSub<G, F, P>,
//!     F: ArgminFloat,
//! {
//!     // This gives the solver a name which will be used for logging
//!     const NAME: &'static str = "Landweber";
//!
//!     // Defines the computations performed in a single iteration.
//!     fn next_iter(
//!         &mut self,
//!         // This gives access to the operator supplied to the `Executor`. `O` implements
//!         // `Gradient` and `OpWrapper` takes care of counting the calls to the respective
//!         // functions.
//!         op: &mut OpWrapper<O>,
//!         // Current state of the optimization. This gives access to the parameter vector,
//!         // gradient, Hessian and cost function value of the current, previous and best
//!         // iteration as well as current iteration number, and many more.
//!         mut state: IterState<P, G, (), (), F>,
//!     ) -> Result<(IterState<P, G, (), (), F>, Option<ArgminKV>), Error> {
//!         // First we obtain the current parameter vector from the `state` struct (`x_k`).
//!         let xk = state.take_param().unwrap();
//!         // Then we compute the gradient at `x_k` (`\nabla f(x_k)`)
//!         let grad = op.gradient(&xk)?;
//!         // Now subtract `\nabla f(x_k)` scaled by `omega` from `x_k` to compute `x_{k+1}`
//!         let xkp1 = xk.scaled_sub(&self.omega, &grad);
//!         // Return new the updated `state`
//!         Ok((state.param(xkp1), None))
//!     }
//! }
//! ```
//!
//!
//! # License
//!
//! Licensed under either of
//!
//!   * Apache License, Version 2.0,
//!     ([LICENSE-APACHE](https://github.com/argmin-rs/argmin/blob/main/LICENSE-APACHE) or
//!     <http://www.apache.org/licenses/LICENSE-2.0>)
//!   * MIT License ([LICENSE-MIT](https://github.com/argmin-rs/argmin/blob/main/LICENSE-MIT) or
//!     <http://opensource.org/licenses/MIT>)
//!
//! at your option.
//!
//! ## Contribution
//!
//! Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion
//! in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above,
//! without any additional terms or conditions.

#![warn(missing_docs)]
#![allow(unused_attributes)]
// Explicitly disallow EQ comparison of floats. (This clippy lint is denied by default; however,
// this is just to make sure that it will always stay this way.)
#![deny(clippy::float_cmp)]

extern crate rand;

#[macro_use]
pub mod core;

/// Solvers
pub mod solver;

/// Macros
#[macro_use]
mod macros;

#[cfg(test)]
#[cfg(feature = "ndarrayl")]
mod tests;
