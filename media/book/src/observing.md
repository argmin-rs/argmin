# Observing progress

Argmin offers an interface to observe the state of the solver at initialization as well as after every iteration.
This includes the parameter vector, gradient, Jacobian, Hessian, iteration number, cost values and many more general as well as solver-specific metrics.
This interface can be used to implement loggers, send the information to a storage or to plot metrics.

The observer [`ParamWriter`](https://docs.rs/argmin-observer-paramwriter/latest/argmin_observer_paramwriter/struct.ParamWriter.html) saves the parameter vector
to disk and as such requires the parameter vector to be serializable.
This observer is available in the [`argmin-observer-paramwriter`](https://crates.io/crates/argmin-observer-paramwriter) crate.

The observer [`SlogLogger`](https://docs.rs/argmin-observer-slog/latest/argmin_observer_slog/struct.SlogLogger.html) logs the progress of the optimization to screen or to disk.
This can be found in the [`argmin-observer-slog`](https://crates.io/crates/argmin-observer-slog) crate.
Writing to disk requires the `serde1` feature to be enabled in `argmin-observer-slog`.

The rate at which the progress of the solver is observed can be set via `ObserverMode`,
which can be either `Always`, `Never`, `NewBest` (whenever a new best solution is found) or `Every(i)` which means every `i`th iteration.

Custom observers can be used as well by implementing the [`Observe`](https://docs.rs/argmin/latest/argmin/core/observers/trait.Observe.html) trait
(see the chapter on [implementing an observer](./implementing_observer.md) for details).

The following example shows how to add an observer to an `Executor` which logs progress to the terminal.
The mode `ObserverMode::Always` ensures that every iteration is printed to screen.
Multiple observers can be added to a single `Executor`.

```rust
# #![allow(unused_imports)]
# extern crate argmin;
# extern crate argmin_testfunctions;
# use argmin::core::{Error, Executor, CostFunction, Gradient};
use argmin::core::observers::ObserverMode;
use argmin_observer_slog::SlogLogger;
# use argmin::solver::gradientdescent::SteepestDescent;
# use argmin::solver::linesearch::MoreThuenteLineSearch;
# use argmin_testfunctions::{rosenbrock, rosenbrock_derivative};
#
# struct Rosenbrock {}
#
# /// Implement `CostFunction` for `Rosenbrock`
# impl CostFunction for Rosenbrock {
#     /// Type of the parameter vector
#     type Param = Vec<f64>;
#     /// Type of the return value computed by the cost function
#     type Output = f64;
#
#     /// Apply the cost function to a parameter `p`
#     fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
#         Ok(rosenbrock(p))
#     }
# }
#
# /// Implement `Gradient` for `Rosenbrock`
# impl Gradient for Rosenbrock {
#     /// Type of the parameter vector
#     type Param = Vec<f64>;
#     /// Type of the return value computed by the cost function
#     type Gradient = Vec<f64>;
#
#     /// Compute the gradient at parameter `p`.
#     fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
#         Ok(rosenbrock_derivative(p))
#     }
# }
#
# fn run() -> Result<(), Error> {
# 
# // Define cost function (must implement `CostFunction` and `Gradient`)
# let cost = Rosenbrock {};
#  
# // Define initial parameter vector
# let init_param: Vec<f64> = vec![-1.2, 1.0];
#  
# // Set up line search
# let linesearch = MoreThuenteLineSearch::new();
#  
# // Set up solver
# let solver = SteepestDescent::new(linesearch);

// [...]

let res = Executor::new(cost, solver)
    .configure(|state| state.param(init_param).max_iters(10))
    // Add an observer which will log all iterations to the terminal
    .add_observer(SlogLogger::term(), ObserverMode::Always)
    .run()?;
#
# println!("{}", res);
#     Ok(())
# }
#
# fn main() {
#     if let Err(ref e) = run() {
#         println!("{}", e);
#         std::process::exit(1);
#     }
# }
```

<!--
## Using Spectator

[Spectator](https://crates.io/crates/spectator)  is a graphical visualization tool for showing the progress of optimization runs.
It is a dedicated program which receives metrics from the observer [argmin-observer-spectator](https://crates.io/crates/argmin-observer-spectator). 

In order to install spectator, run

```shell
cargo install spectator --locked
```

To start spectator, run

```shell
spectator
```

This will start a server which binds to `0.0.0.0:5498`. To change this, provide `--host` and `--port`:

```shell
spectator --host 127.0.0.1 --port 1234
```

Once spectator started, it will wait for data on the provided address. 
All that needs to be done now is to add the observer to the optimization:

```rust
# #![allow(unused_imports)]
# extern crate argmin;
# extern crate argmin_testfunctions;
# use argmin::core::{Error, Executor, CostFunction, Gradient};
use argmin::core::observers::ObserverMode;
use argmin_observer_spectator::SpectatorBuilder;
# use argmin::solver::gradientdescent::SteepestDescent;
# use argmin::solver::linesearch::MoreThuenteLineSearch;
# use argmin_testfunctions::{rosenbrock, rosenbrock_derivative};
#
# struct Rosenbrock {}
#
# /// Implement `CostFunction` for `Rosenbrock`
# impl CostFunction for Rosenbrock {
#     /// Type of the parameter vector
#     type Param = Vec<f64>;
#     /// Type of the return value computed by the cost function
#     type Output = f64;
#
#     /// Apply the cost function to a parameter `p`
#     fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
#         Ok(rosenbrock(p))
#     }
# }
#
# /// Implement `Gradient` for `Rosenbrock`
# impl Gradient for Rosenbrock {
#     /// Type of the parameter vector
#     type Param = Vec<f64>;
#     /// Type of the return value computed by the cost function
#     type Gradient = Vec<f64>;
#
#     /// Compute the gradient at parameter `p`.
#     fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
#         Ok(rosenbrock_derivative(p))
#     }
# }
#
# fn run() -> Result<(), Error> {
# 
# // Define cost function (must implement `CostFunction` and `Gradient`)
# let cost = Rosenbrock {};
#  
# // Define initial parameter vector
# let init_param: Vec<f64> = vec![-1.2, 1.0];
#  
# // Set up line search
# let linesearch = MoreThuenteLineSearch::new();
#  
# // Set up solver
# let solver = SteepestDescent::new(linesearch);

// [...]

let spectator = SpectatorBuilder::new()
    // Optional: Defaults to 0.0.0.0
    .with_host("127.0.0.1")
    // Optional: Defaults to 5498
    .with_port(1234)
    // Optionally give optimization a name.
    // If not provided a random UUID will be used
    .with_name("something")
    // Optionally select a subset of the available metrics.
    // If omitted, all metrics will be selected.
    // Note that still all metrics are sent to spectator,
    // however; only those selected will be shown.
    // Spectator allows to select metrics in the GUI as well.
    .select(&["cost", "best_cost", "t"])
    .build();


let res = Executor::new(cost, solver)
    .configure(|state| state.param(init_param).max_iters(10))
    .add_observer(spectator, ObserverMode::Always)
    .run()?;
#
# println!("{}", res);
#     Ok(())
# }
#
# fn main() {
#     if let Err(ref e) = run() {
#         println!("{}", e);
#         std::process::exit(1);
#     }
# }
```
-->
