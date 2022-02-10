# argmin
[![argmin CI](https://github.com/argmin-rs/argmin/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/argmin-rs/argmin/actions/workflows/ci.yml)
[![argmin on deps.rs](https://deps.rs/repo/github/argmin-rs/argmin/status.svg)](https://deps.rs/repo/github/argmin-rs/argmin)
[![argmin on crates.io](https://img.shields.io/crates/v/argmin)](https://crates.io/crates/argmin)
[![argmin on docs.rs](https://docs.rs/argmin/badge.svg)](https://docs.rs/argmin)
[![Source Code Repository](https://img.shields.io/badge/Code-On%20github.com-blue)](https://github.com/argmin-rs/argmin)
![Maintenance](https://img.shields.io/badge/maintenance-actively--developed-brightgreen.svg)
![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)
[![Gitter chat](https://badges.gitter.im/argmin-rs/community.png)](https://gitter.im/argmin-rs/community)

argmin is a numerical optimization library written entirely in Rust.

[Documentation of most recent release][__link0]

[Documentation of main branch][__link1]

**This is the README for the current development version. For the README of the most recent release please visit [crates.io](https://crates.io/crates/argmin)!**

## Design goals

argmin aims at offering a wide range of optimization algorithms with a consistent interface, written purely in Rust. It comes with additional features such as checkpointing and observers which for instance make it possible to log the progress of an optimization to screen or file.

It further provides a framework for implementing iterative optimization algorithms in a convenient manner. Essentially, a single iteration of the algorithm needs to be implemented and everything else, such as handling termination, parameter vectors, gradients and Hessians, is taken care of by the library.

This library uses generics to be as type-agnostic as possible. Abstractions over common math functions enable the use of common backends such as `ndarray` and `nalgebra` via the `argmin-math` crate. All operations can be performed with 32 and 64 bit floats. Custom types are of course also supported.


## Contributing

This crate is looking for contributors! Potential projects can be found in the [Github issues][__link2], but feel free to suggest your own ideas as well. Besides adding optimization methods and new features, other contributions are also highly welcome, for instance improving performance, documentation, writing examples (with real world problems), developing tests, adding observers, implementing a C interface or [Python wrappers][__link3]. Bug reports (and fixes) are of course also highly appreciated.


## Algorithms

 - [Line searches][__link4]
	
	 - [Backtracking line search][__link5]
	 - [More-Thuente line search][__link6]
	 - [Hager-Zhang line search][__link7]
	
	
 - [Trust region method][__link8]
	
	 - [Cauchy point method][__link9]
	 - [Dogleg method][__link10]
	 - [Steihaug method][__link11]
	
	
 - [Steepest descent][__link12]
	
	
 - [Conjugate gradient method][__link13]
	
	
 - [Nonlinear conjugate gradient method][__link14]
	
	
 - [Newton methods][__link15]
	
	 - [Newton’s method][__link16]
	 - [Newton-CG][__link17]
	
	
 - [Quasi-Newton methods][__link18]
	
	 - [BFGS][__link19]
	 - [L-BFGS][__link20]
	 - [DFP][__link21]
	 - [SR1][__link22]
	 - [SR1-TrustRegion][__link23]
	
	
 - [Gauss-Newton method][__link24]
	
	
 - [Gauss-Newton method with linesearch][__link25]
	
	
 - [Golden-section search][__link26]
	
	
 - [Landweber iteration][__link27]
	
	
 - [Brent’s method][__link28]
	
	
 - [Nelder-Mead method][__link29]
	
	
 - [Simulated Annealing][__link30]
	
	
 - [Particle Swarm Optimization][__link31]
	
	


## Examples

Examples for each solver can be found [here (current released version)][__link32] and [here (main branch)][__link33].


## Usage

Add this to your `Cargo.toml`:


```toml
[dependencies]
argmin = "0.5.0"
argmin-math = { version = "0.1.0", features = ["ndarray_latest-serde,nalgebra_latest-serde"] }
```

or, for the current development version:


```toml
[dependencies]
argmin = { git = "https://github.com/argmin-rs/argmin" }
argmin-math = { git = "https://github.com/argmin-rs/argmin", features = ["ndarray_latest-serde,nalgebra_latest-serde"] }
```

(For which features to select for `argmin-math` please see the [documentation][__link34].)


### Features


#### Default features

 - `slog-logger`: Support for logging using `slog`
 - `serde1`: Support for `serde`. Needed for checkpointing and writing parameters to disk as well as logging to disk.


#### Optional features

The `ctrlc` feature uses the `ctrlc` crate to properly stop the optimization (and return the current best result) after pressing Ctrl+C during an optimization run.


```toml
[dependencies]
argmin = { version = "0.5.0", features = ["ctrlc"] }
```


#### Experimental support for compiling to WebAssembly

When compiling to WASM, the feature `wasm-bindgen` must be used.

WASM support is still experimental. Please report any issues you encounter when using argmin in a WASM context.


#### Compiling without `serde` dependency

The `serde` dependency can be removed by turning off the `serde1` feature, for instance like so:


```toml
[dependencies]
argmin = { version = "0.5.0", default-features = false, features = ["slog-logger"] }
```

Note that this will remove the ability to write parameters and logs to disk as well as checkpointing.


### Running the tests and building the examples

The tests and examples require a set of features to be enabled:


```bash
cargo test --features "argmin/ctrlc,argmin-math/ndarray_latest-serde,argmin-math/nalgebra_latest-serde,argmin/ndarrayl"
```


## Defining a problem

A problem can be defined by implementing the `ArgminOp` trait which comes with the associated types `Param`, `Output` and `Hessian`. `Param` is the type of your parameter vector (i.e. the input to your cost function), `Output` is the type returned by the cost function, `Hessian` is the type of the Hessian and `Jacobian` is the type of the Jacobian. The trait provides the following methods:

 - `apply(&self, p: &Self::Param) -> Result<Self::Output, Error>`: Applys the cost function to parameters `p` of type `Self::Param` and returns the cost function value.
 - `gradient(&self, p: &Self::Param) -> Result<Self::Param, Error>`: Computes the gradient at `p`.
 - `hessian(&self, p: &Self::Param) -> Result<Self::Hessian, Error>`: Computes the Hessian at `p`.
 - `jacobian(&self, p: &Self::Param) -> Result<Self::Jacobian, Error>`: Computes the Jacobian at `p`.

The following code snippet shows an example of how to use the Rosenbrock test functions from `argmin-testfunctions` in argmin:


```rust
use argmin_testfunctions::{rosenbrock_2d, rosenbrock_2d_derivative, rosenbrock_2d_hessian};
use argmin::core::{ArgminOp, Error};

/// First, create a struct for your problem
struct Rosenbrock {
    a: f64,
    b: f64,
}

/// Implement `ArgminOp` for `Rosenbrock`
impl ArgminOp for Rosenbrock {
    /// Type of the parameter vector
    type Param = Vec<f64>;
    /// Type of the return value computed by the cost function
    type Output = f64;
    /// Type of the Hessian. Can be `()` if not needed.
    type Hessian = Vec<Vec<f64>>;
    /// Type of the Jacobian. Can be `()` if not needed.
    type Jacobian = ();
    /// Floating point precision
    type Float = f64;

    /// Apply the cost function to a parameter `p`
    fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        Ok(rosenbrock_2d(p, self.a, self.b))
    }

    /// Compute the gradient at parameter `p`.
    fn gradient(&self, p: &Self::Param) -> Result<Self::Param, Error> {
        Ok(rosenbrock_2d_derivative(p, self.a, self.b))
    }

    /// Compute the Hessian at parameter `p`.
    fn hessian(&self, p: &Self::Param) -> Result<Self::Hessian, Error> {
        let t = rosenbrock_2d_hessian(p, self.a, self.b);
        Ok(vec![vec![t[0], t[1]], vec![t[2], t[3]]])
    }
}
```

It is optional to implement any of these methods, as there are default implementations which will return an `Err` when called. What needs to be implemented is defined by the requirements of the solver that is to be used.


## Running a solver

The following example shows how to use the previously shown definition of a problem in a Steepest Descent (Gradient Descent) solver.


```rust
use argmin::core::{ArgminOp, Error, Executor};
use argmin::core::{ArgminSlogLogger, ObserverMode};
use argmin::solver::gradientdescent::SteepestDescent;
use argmin::solver::linesearch::MoreThuenteLineSearch;

// Define cost function (must implement `ArgminOperator`)
let cost = Rosenbrock { a: 1.0, b: 100.0 };
 
// Define initial parameter vector
let init_param: Vec<f64> = vec![-1.2, 1.0];
 
// Set up line search
let linesearch = MoreThuenteLineSearch::new();
 
// Set up solver
let solver = SteepestDescent::new(linesearch);
 
// Run solver
let res = Executor::new(cost, solver, init_param)
    // Add an observer which will log all iterations to the terminal
    .add_observer(ArgminSlogLogger::term(), ObserverMode::Always)
    // Set maximum iterations to 10
    .max_iters(10)
    // run the solver on the defined problem
    .run()?;
// print result
println!("{}", res);
```


## Observing iterations

Argmin offers an interface to observe the state of the solver at initialization as well as after every iteration. This includes the parameter vector, gradient, Hessian, iteration number, cost values and many more as well as solver-specific metrics. This interface can be used to implement loggers, send the information to a storage or to plot metrics. Observers need to implement the `Observe` trait. Argmin ships with a logger based on the `slog` crate. `ArgminSlogLogger::term` logs to the terminal and `ArgminSlogLogger::file` logs to a file in JSON format. Both loggers also come with a `*_noblock` version which does not block the execution of logging, but may drop some messages in case of a full buffer. Parameter vectors can be written to disk using `WriteToFile`. For each observer it can be defined how often it will observe the progress of the solver. This is indicated via the enum `ObserverMode` which can be either `Always`, `Never`, `NewBest` (whenever a new best solution is found) or `Every(i)` which means every `i`th iteration.


```rust
let res = Executor::new(problem, solver, init_param)
let res = res
    // Add an observer which will log all iterations to the terminal (without blocking)
    .add_observer(ArgminSlogLogger::term_noblock(), ObserverMode::Always)
    // Log to file whenever a new best solution is found
    .add_observer(ArgminSlogLogger::file("solver.log", false)?, ObserverMode::NewBest)
    // Write parameter vector to `params/param.arg` every 20th iteration
    .add_observer(WriteToFile::new("params", "param"), ObserverMode::Every(20))
    // run the solver on the defined problem
    .run()?;
```


## Checkpoints

The probability of crashes increases with runtime, therefore one may want to save checkpoints in order to be able to resume the optimization after a crash. The `CheckpointMode` defines how often checkpoints are saved and is either `Never` (default), `Always` (every iteration) or `Every(u64)` (every Nth iteration). It is set via the setter method `checkpoint_mode` of `Executor`. In addition, the directory where the checkpoints and a prefix for every file can be set via `checkpoint_dir` and `checkpoint_name`, respectively.

The following example shows how the `from_checkpoint` method can be used to resume from a checkpoint. In case this fails (for instance because the file does not exist, which could mean that this is the first run and there is nothing to resume from), it will resort to creating a new `Executor`, thus starting from scratch.


```rust
let res = Executor::from_checkpoint(".checkpoints/optim.arg", Rosenbrock {})
    .unwrap_or(Executor::new(Rosenbrock {}, solver, init_param))
    .max_iters(iters)
    .checkpoint_dir(".checkpoints")
    .checkpoint_name("optim")
    .checkpoint_mode(CheckpointMode::Every(20))
    .run()?;
```


## Implementing an optimization algorithm

In this section we are going to implement the Landweber solver, which essentially is a special form of gradient descent. In iteration `k`, the new parameter vector `x_{k+1}` is calculated from the previous parameter vector `x_k` and the gradient at `x_k` according to the following update rule:

`x_{k+1} = x_k - omega * \nabla f(x_k)`

In order to implement this using the argmin framework, one first needs to define a struct which holds data specific to the solver. Then, the `Solver` trait needs to be implemented for the struct. This requires setting the associated constant `NAME` which gives your solver a name. The `next_iter` method defines the computations performed in a single iteration of the solver. Via the parameters `op` and `state` one has access to the operator (cost function, gradient computation, Hessian, …) and to the current state of the optimization (parameter vectors, cost function values, iteration number, …), respectively.


```rust
use argmin::core::{ArgminFloat, ArgminIterData, ArgminOp, Error, IterState, OpWrapper, Solver};
#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};
use argmin_math::ArgminScaledSub;

// Define a struct which holds any parameters/data which are needed during the execution of the
// solver. Note that this does not include parameter vectors, gradients, Hessians, cost
// function values and so on, as those will be handled by the `Executor`.
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct Landweber<F> {
    /// omega
    omega: F,
}

impl<F> Landweber<F> {
    /// Constructor
    pub fn new(omega: F) -> Self {
        Landweber { omega }
    }
}

impl<O, F> Solver<O> for Landweber<F>
where
    // `O` always needs to implement `ArgminOp`
    O: ArgminOp<Float = F>,
    // `O::Param` needs to implement `ArgminScaledSub` because of the update formula
    O::Param: ArgminScaledSub<O::Param, O::Float, O::Param>,
    F: ArgminFloat,
{
    // This gives the solver a name which will be used for logging
    const NAME: &'static str = "Landweber";

    // Defines the computations performed in a single iteration.
    fn next_iter(
        &mut self,
        // This gives access to the operator supplied to the `Executor`. `O` implements
        // `ArgminOp` and `OpWrapper` takes care of counting the calls to the respective
        // functions.
        op: &mut OpWrapper<O>,
        // Current state of the optimization. This gives access to the parameter vector,
        // gradient, Hessian and cost function value of the current, previous and best
        // iteration as well as current iteration number, and many more.
        state: &IterState<O>,
    ) -> Result<ArgminIterData<O>, Error> {
        // First we obtain the current parameter vector from the `state` struct (`x_k`).
        let xk = state.get_param();
        // Then we compute the gradient at `x_k` (`\nabla f(x_k)`)
        let grad = op.gradient(&xk)?;
        // Now subtract `\nabla f(x_k)` scaled by `omega` from `x_k` to compute `x_{k+1}`
        let xkp1 = xk.scaled_sub(&self.omega, &grad);
        // Return new paramter vector which will then be used by the `Executor` to update
        // `state`.
        Ok(ArgminIterData::new().param(xkp1))
    }
}
```


## License

Licensed under either of

 - Apache License, Version 2.0, ([LICENSE-APACHE][__link35] or <http://www.apache.org/licenses/LICENSE-2.0>)
 - MIT License ([LICENSE-MIT][__link37] or <http://opensource.org/licenses/MIT>)

at your option.


### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.


 [__link0]: https://docs.rs/argmin/latest/argmin/
 [__link1]: https://argmin-rs.github.io/argmin/argmin/
 [__link10]: https://argmin-rs.github.io/argmin/argmin/solver/trustregion/dogleg/struct.Dogleg.html
 [__link11]: https://argmin-rs.github.io/argmin/argmin/solver/trustregion/steihaug/struct.Steihaug.html
 [__link12]: https://argmin-rs.github.io/argmin/argmin/solver/gradientdescent/steepestdescent/struct.SteepestDescent.html
 [__link13]: https://argmin-rs.github.io/argmin/argmin/solver/conjugategradient/cg/struct.ConjugateGradient.html
 [__link14]: https://argmin-rs.github.io/argmin/argmin/solver/conjugategradient/nonlinear_cg/struct.NonlinearConjugateGradient.html
 [__link15]: https://argmin-rs.github.io/argmin/argmin/solver/newton/index.html
 [__link16]: https://argmin-rs.github.io/argmin/argmin/solver/newton/newton_method/struct.Newton.html
 [__link17]: https://argmin-rs.github.io/argmin/argmin/solver/newton/newton_cg/struct.NewtonCG.html
 [__link18]: https://argmin-rs.github.io/argmin/argmin/solver/quasinewton/index.html
 [__link19]: https://argmin-rs.github.io/argmin/argmin/solver/quasinewton/bfgs/struct.BFGS.html
 [__link2]: https://github.com/argmin-rs/argmin/issues
 [__link20]: https://argmin-rs.github.io/argmin/argmin/solver/quasinewton/lbfgs/struct.LBFGS.html
 [__link21]: https://argmin-rs.github.io/argmin/argmin/solver/quasinewton/dfp/struct.DFP.html
 [__link22]: https://argmin-rs.github.io/argmin/argmin/solver/quasinewton/sr1/struct.SR1.html
 [__link23]: https://argmin-rs.github.io/argmin/argmin/solver/quasinewton/sr1_trustregion/struct.SR1TrustRegion.html
 [__link24]: https://argmin-rs.github.io/argmin/argmin/solver/gaussnewton/gaussnewton_method/struct.GaussNewton.html
 [__link25]: https://argmin-rs.github.io/argmin/argmin/solver/gaussnewton/gaussnewton_linesearch/struct.GaussNewtonLS.html
 [__link26]: https://argmin-rs.github.io/argmin/argmin/solver/goldensectionsearch/struct.GoldenSectionSearch.html
 [__link27]: https://argmin-rs.github.io/argmin/argmin/solver/landweber/struct.Landweber.html
 [__link28]: https://argmin-rs.github.io/argmin/argmin/solver/brent/struct.Brent.html
 [__link29]: https://argmin-rs.github.io/argmin/argmin/solver/neldermead/struct.NelderMead.html
 [__link3]: https://github.com/argmin-rs/pyargmin
 [__link30]: https://argmin-rs.github.io/argmin/argmin/solver/simulatedannealing/struct.SimulatedAnnealing.html
 [__link31]: https://argmin-rs.github.io/argmin/argmin/solver/particleswarm/struct.ParticleSwarm.html
 [__link32]: https://github.com/argmin-rs/argmin/tree/v0.5.0/examples
 [__link33]: https://github.com/argmin-rs/argmin/tree/main/argmin/examples
 [__link34]: https://docs.rs/argmin/latest/argmin-math
 [__link35]: https://github.com/argmin-rs/argmin/blob/main/LICENSE-APACHE
 [__link37]: https://github.com/argmin-rs/argmin/blob/main/LICENSE-MIT
 [__link4]: https://argmin-rs.github.io/argmin/argmin/solver/linesearch/index.html
 [__link5]: https://argmin-rs.github.io/argmin/argmin/solver/linesearch/backtracking/struct.BacktrackingLineSearch.html
 [__link6]: https://argmin-rs.github.io/argmin/argmin/solver/linesearch/morethuente/struct.MoreThuenteLineSearch.html
 [__link7]: https://argmin-rs.github.io/argmin/argmin/solver/linesearch/hagerzhang/struct.HagerZhangLineSearch.html
 [__link8]: https://argmin-rs.github.io/argmin/argmin/solver/trustregion/trustregion_method/struct.TrustRegion.html
 [__link9]: https://argmin-rs.github.io/argmin/argmin/solver/trustregion/cauchypoint/struct.CauchyPoint.html
