# Observing progress

Argmin offers an interface to observe the state of the solver at initialization as well as after every iteration.
This includes the parameter vector, gradient, Jacobian, Hessian, iteration number, cost values and many more as well as solver-specific metrics.
This interface can be used to implement loggers, send the information to a storage or to plot metrics.

The observer [`WriteToFile`](https://docs.rs/argmin/latest/argmin/core/observers/file/struct.WriteToFile.html) saves the parameter vector to disk and as such requires the parameter vector to be serializable.
Hence this feature is only available with the `serde1` feature.

The observer [`SlogLogger`](https://docs.rs/argmin/latest/argmin/core/observers/slog_logger/struct.SlogLogger.html) logs the progress of the optimization to screen or to disk.
This requires the `slog-logger` feature.
Writing to disk in addtion requires the `serde1` feature.

For each observer it can be defined how often it will observe the progress of the solver.
This is indicated via the enum `ObserverMode` which can be either `Always`, `Never`, `NewBest` (whenever a new best solution is found) or `Every(i)` which means every `i`th iteration.

Custom observers can be used as well by implementing the [`Observe`](https://docs.rs/argmin/latest/argmin/core/observers/trait.Observe.html) trait (see the chapter on [implementing an observer](./implementing_observer.md) for details).

The following example shows how to add an observer to an `Executor` which logs progress to the terminal.
The observer is configured via `ObserverMode::Always` such that it will log every iteration to screen.
Multiple observers can be added to a single `Executor`.

```rust
# #![allow(unused_imports)]
# extern crate argmin;
# extern crate argmin_testfunctions;
# use argmin::core::{Error, Executor, CostFunction, Gradient};
use argmin::core::observers::{SlogLogger, ObserverMode};
# use argmin::solver::gradientdescent::SteepestDescent;
# use argmin::solver::linesearch::MoreThuenteLineSearch;
# use argmin_testfunctions::{rosenbrock_2d, rosenbrock_2d_derivative};
#
# struct Rosenbrock {
#     a: f64,
#     b: f64,
# }
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
#         Ok(rosenbrock_2d(p, 1.0, 100.0))
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
#         Ok(rosenbrock_2d_derivative(p, 1.0, 100.0))
#     }
# }
#
# fn run() -> Result<(), Error> {
# 
# // Define cost function (must implement `CostFunction` and `Gradient`)
# let cost = Rosenbrock { a: 1.0, b: 100.0 };
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
