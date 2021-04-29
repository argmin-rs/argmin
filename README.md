[![CircleCI](https://circleci.com/gh/argmin-rs/argmin.svg?style=svg)](https://circleci.com/gh/argmin-rs/argmin)
[![Build Status](https://travis-ci.org/argmin-rs/argmin.svg?branch=master)](https://travis-ci.org/argmin-rs/argmin)
[![Gitter chat](https://badges.gitter.im/argmin-rs/community.png)](https://gitter.im/argmin-rs/community)

# argmin

A pure Rust optimization framework

This crate offers a numerical optimization toolbox/framework written entirely
in Rust. It is at the moment potentially very buggy. Please use with care and report any bugs
you encounter. This crate is looking for contributors!

[Documentation of most recent release](https://docs.rs/argmin/latest/argmin/)

[Documentation of master](https://argmin-rs.github.io/argmin/argmin/)

## Design goals

This crate's intention is to be useful to users as well as developers of optimization
algorithms, meaning that it should be both easy to apply and easy to implement algorithms. In
particular, as a developer of optimization algorithms you should not need to worry about
usability features (such as logging, dealing with different types, setters and getters for
certain common parameters, counting cost function and gradient evaluations, termination, and so
on). Instead you can focus on implementing your algorithm.

- Easy framework for the implementation of optimization algorithms: Implement a single iteration
  of your method and let the framework do the rest. This leads to similar interfaces for
  different solvers, making it easy for users.
- Pure Rust implementations of a wide range of optimization methods: This avoids the need to
  compile and interface C/C++/Fortran code.
- Type-agnostic: Many problems require data structures that go beyond simple vectors to
  represent the parameters. In argmin, everything is generic: All that needs to be done is
  implementing certain traits on your data type. For common types, these traits are already
  implemented.
- Convenient: Easy and consistent logging of anything that may be important. Log to the
  terminal, to a file or implement your own observers. Future plans include sending metrics to
  databases and connecting to big data piplines.
- Algorithm evaluation: Methods to assess the performance of an algorithm for different
  parameter settings, problem classes, ...

Since this crate is in a very early stage, so far most points are only partially implemented or
remain future plans.

## Algorithms

- [Line searches](https://argmin-rs.github.io/argmin/argmin/solver/linesearch/index.html)
  - [Backtracking line search](https://argmin-rs.github.io/argmin/argmin/solver/linesearch/backtracking/struct.BacktrackingLineSearch.html)
  - [More-Thuente line search](https://argmin-rs.github.io/argmin/argmin/solver/linesearch/morethuente/struct.MoreThuenteLineSearch.html)
  - [Hager-Zhang line search](https://argmin-rs.github.io/argmin/argmin/solver/linesearch/hagerzhang/struct.HagerZhangLineSearch.html)
- [Trust region method](https://argmin-rs.github.io/argmin/argmin/solver/trustregion/trustregion_method/struct.TrustRegion.html)
  - [Cauchy point method](https://argmin-rs.github.io/argmin/argmin/solver/trustregion/cauchypoint/struct.CauchyPoint.html)
  - [Dogleg method](https://argmin-rs.github.io/argmin/argmin/solver/trustregion/dogleg/struct.Dogleg.html)
  - [Steihaug method](https://argmin-rs.github.io/argmin/argmin/solver/trustregion/steihaug/struct.Steihaug.html)
- [Steepest descent](https://argmin-rs.github.io/argmin/argmin/solver/gradientdescent/steepestdescent/struct.SteepestDescent.html)
- [Conjugate gradient method](https://argmin-rs.github.io/argmin/argmin/solver/conjugategradient/cg/struct.ConjugateGradient.html)
- [Nonlinear conjugate gradient method](https://argmin-rs.github.io/argmin/argmin/solver/conjugategradient/nonlinear_cg/struct.NonlinearConjugateGradient.html)
- [Newton methods](https://argmin-rs.github.io/argmin/argmin/solver/newton/index.html)
  - [Newton's method](https://argmin-rs.github.io/argmin/argmin/solver/newton/newton_method/struct.Newton.html)
  - [Newton-CG](https://argmin-rs.github.io/argmin/argmin/solver/newton/newton_cg/struct.NewtonCG.html)
- [Quasi-Newton methods](https://argmin-rs.github.io/argmin/argmin/solver/quasinewton/index.html)
  - [BFGS](https://argmin-rs.github.io/argmin/argmin/solver/quasinewton/bfgs/struct.BFGS.html)
  - [L-BFGS](https://argmin-rs.github.io/argmin/argmin/solver/quasinewton/lbfgs/struct.LBFGS.html)
  - [DFP](https://argmin-rs.github.io/argmin/argmin/solver/quasinewton/dfp/struct.DFP.html)
  - [SR1](https://argmin-rs.github.io/argmin/argmin/solver/quasinewton/sr1/struct.SR1.html)
  - [SR1-TrustRegion](https://argmin-rs.github.io/argmin/argmin/solver/quasinewton/sr1_trustregion/struct.SR1TrustRegion.html)
- [Gauss-Newton method](https://argmin-rs.github.io/argmin/argmin/solver/gaussnewton/gaussnewton_method/struct.GaussNewton.html)
- [Gauss-Newton method with linesearch](https://argmin-rs.github.io/argmin/argmin/solver/gaussnewton/gaussnewton_linesearch/struct.GaussNewtonLS.html)
- [Golden-section search](https://argmin-rs.github.io/argmin/argmin/solver/goldensectionsearch/struct.GoldenSectionSearch.html)
- [Landweber iteration](https://argmin-rs.github.io/argmin/argmin/solver/landweber/struct.Landweber.html)
- [Brent's method](https://argmin-rs.github.io/argmin/argmin/solver/brent/index.html)
- [Nelder-Mead method](https://argmin-rs.github.io/argmin/argmin/solver/neldermead/struct.NelderMead.html)
- [Simulated Annealing](https://argmin-rs.github.io/argmin/argmin/solver/simulatedannealing/struct.SimulatedAnnealing.html)
- [Particle Swarm Optimization](https://argmin-rs.github.io/argmin/argmin/solver/particleswarm/struct.ParticleSwarm.html)

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
argmin = "0.4.4"
```

### Optional features (recommended)

There are additional features which can be activated in `Cargo.toml`:

```toml
[dependencies]
argmin = { version = "0.4.4", features = ["ctrlc", "ndarrayl", "nalgebral"] }
```

These may become default features in the future. Without these features compilation to
`wasm32-unknown-unkown` seems to be possible.

- `ctrlc`: Uses the `ctrlc` crate to properly stop the optimization (and return the current best
   result) after pressing Ctrl+C.
- `ndarrayl`: Support for `ndarray`, `ndarray-linalg` and `ndarray-rand`.
- `nalgebral`: Support for [`nalgebra`](https://nalgebra.org).

### Running the tests

Running the tests requires the `ndarrayl` and `nalgebral` features to be enabled

```bash
cargo test --features "ndarrayl nalgebral"
```

## Defining a problem

A problem can be defined by implementing the `ArgminOp` trait which comes with the
associated types `Param`, `Output` and `Hessian`. `Param` is the type of your
parameter vector (i.e. the input to your cost function), `Output` is the type returned
by the cost function and `Hessian` is the type of the Hessian.
The trait provides the following methods:

- `apply(&self, p: &Self::Param) -> Result<Self::Output, Error>`: Applys the cost
  function to parameters `p` of type `Self::Param` and returns the cost function value.
- `gradient(&self, p: &Self::Param) -> Result<Self::Param, Error>`: Computes the
  gradient at `p`.
- `hessian(&self, p: &Self::Param) -> Result<Self::Hessian, Error>`: Computes the Hessian
  at `p`.

The following code snippet shows an example of how to use the Rosenbrock test functions from
`argmin-testfunctions` in argmin:

```rust
use argmin::testfunctions::{rosenbrock_2d, rosenbrock_2d_derivative, rosenbrock_2d_hessian};
use argmin::prelude::*;
use serde::{Serialize, Deserialize};

/// First, create a struct for your problem
#[derive(Clone, Default, Serialize, Deserialize)]
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

It is optional to implement any of these methods, as there are default implementations which
will return an `Err` when called. What needs to be implemented is defined by the requirements
of the solver that is to be used.

## Running a solver

The following example shows how to use the previously shown definition of a problem in a
Steepest Descent (Gradient Descent) solver.

```rust
use argmin::prelude::*;
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

Argmin offers an interface to observe the state of the iteration at initialization as well as
after every iteration. This includes the parameter vector, gradient, Hessian, iteration number,
cost values and many more as well as solver-specific metrics. This interface can be used to
implement loggers, send the information to a storage or to plot metrics.
Observers need to implment the `Observe` trait.
Argmin ships with a logger based on the `slog` crate. `ArgminSlogLogger::term` logs to the
terminal and `ArgminSlogLogger::file` logs to a file in JSON format. Both loggers also come
with a `*_noblock` version which does not block the execution of logging, but may drop some
messages if the buffer is full.
Parameter vectors can be written to disc using `WriteToFile`.
For each observer it can be defined how often it will observe the progress of the solver. This
is indicated via the enum `ObserverMode` which can be either `Always`, `Never`, `NewBest`
(whenever a new best solution is found) or `Every(i)` which means every `i`th iteration.

```rust
let res = Executor::new(problem, solver, init_param)
    // Add an observer which will log all iterations to the terminal (without blocking)
    .add_observer(ArgminSlogLogger::term_noblock(), ObserverMode::Always)
    // Log to file whenever a new best solution is found
    .add_observer(ArgminSlogLogger::file("solver.log")?, ObserverMode::NewBest)
    // Write parameter vector to `params/param.arg` every 20th iteration
    .add_observer(WriteToFile::new("params", "param"), ObserverMode::Every(20))
    // run the solver on the defined problem
    .run()?;
```

## Checkpoints

The probability of crashes increases with runtime, therefore one may want to save checkpoints
in order to be able to resume the optimization after a crash.
The `CheckpointMode` defines how often checkpoints are saved and is either `Never` (default),
`Always` (every iteration) or `Every(u64)` (every Nth iteration). It is set via the setter
method `checkpoint_mode` of `Executor`.
In addition, the directory where the checkpoints and a prefix for every file can be set via
`checkpoint_dir` and `checkpoint_name`, respectively.

The following example shows how the `from_checkpoint` method can be used to resume from a
checkpoint. In case this fails (for instance because the file does not exist, which could mean
that this is the first run and there is nothing to resume from), it will resort to creating a
new `Executor`, thus starting from scratch.

```rust
let res = Executor::from_checkpoint(".checkpoints/optim.arg")
    .unwrap_or(Executor::new(operator, solver, init_param))
    .max_iters(iters)
    .checkpoint_dir(".checkpoints")
    .checkpoint_name("optim")
    .checkpoint_mode(CheckpointMode::Every(20))
    .run()?;
```

## Implementing an optimization algorithm

In this section we are going to implement the Landweber solver, which essentially is a special
form of gradient descent. In iteration `k`, the new parameter vector `x_{k+1}` is calculated
from the previous parameter vector `x_k` and the gradient at `x_k` according to the following
update rule:

`x_{k+1} = x_k - omega * \nabla f(x_k)`

In order to implement this using the argmin framework, one first needs to define a struct which
holds data specific to the solver. Then, the `Solver` trait needs to be implemented for the
struct. This requires setting the associated constant `NAME` which gives your solver a name.
The `next_iter` method defines the computations performed in a single iteration of the solver.
Via the parameters `op` and `state` one has access to the operator (cost function, gradient
computation, Hessian, ...) and to the current state of the optimization (parameter vectors,
cost function values, iteration number, ...), respectively.

```rust
use argmin::prelude::*;
use serde::{Deserialize, Serialize};

// Define a struct which holds any parameters/data which are needed during the execution of the
// solver. Note that this does not include parameter vectors, gradients, Hessians, cost
// function values and so on, as those will be handled by the `Executor`.
#[derive(Serialize, Deserialize)]
pub struct Landweber {
    /// omega
    omega: f64,
}

impl Landweber {
    /// Constructor
    pub fn new(omega: f64) -> Self {
        Landweber { omega }
    }
}

impl<O> Solver<O> for Landweber
where
    // `O` always needs to implement `ArgminOp`
    O: ArgminOp,
    // `O::Param` needs to implement `ArgminScaledSub` because of the update formula
    O::Param: ArgminScaledSub<O::Param, f64, O::Param>,
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

## TODOs

  * More optimization methods
  * Automatic differentiation
  * Parallelization
  * Tests
  * Evaluation on real problems
  * Evaluation framework
  * Documentation & Tutorials
  * C interface
  * Python wrapper
  * Solver and problem definition via a config file

Please open an [issue](https://github.com/argmin-rs/argmin/issues) if you want to contribute!
Any help is appreciated!

## License

Licensed under either of

  * Apache License, Version 2.0,
    ([LICENSE-APACHE](https://github.com/argmin-rs/argmin/blob/master/LICENSE-APACHE) or
    http://www.apache.org/licenses/LICENSE-2.0)
  * MIT License ([LICENSE-MIT](https://github.com/argmin-rs/argmin/blob/master/LICENSE-MIT) or
    http://opensource.org/licenses/MIT)

at your option.


### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion
in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above,
without any additional terms or conditions.

License: MIT OR Apache-2.0
