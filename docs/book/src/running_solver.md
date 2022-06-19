# Running a solver

The following example shows how to use the previously shown definition of a problem in a
Steepest Descent (Gradient Descent) solver.

```rust
# #![allow(unused_imports)]
# extern crate argmin;
# extern crate argmin_testfunctions;
use argmin::core::{Error, Executor, CostFunction, Gradient};
# #[cfg(feature = "slog-logger")]
use argmin::core::observers::{SlogLogger, ObserverMode};
use argmin::solver::gradientdescent::SteepestDescent;
use argmin::solver::linesearch::MoreThuenteLineSearch;
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

// Define cost function (must implement `CostFunction` and `Gradient`)
let cost = Rosenbrock { a: 1.0, b: 100.0 };
 
// Define initial parameter vector
let init_param: Vec<f64> = vec![-1.2, 1.0];
 
// Set up line search
let linesearch = MoreThuenteLineSearch::new();
 
// Set up solver
let solver = SteepestDescent::new(linesearch);
 
// Run solver
let res = Executor::new(cost, solver)
    .configure(|config|
        config
            // Set initial parameters
            .param(init_param)
            // Set maximum iterations to 10
            .max_iters(10)
    )
# ;
# #[cfg(feature = "slog-logger")]
# let res = res
    // Add an observer which will log all iterations to the terminal
    .add_observer(SlogLogger::term(), ObserverMode::Always)
# ;
# let res = res
    // run the solver on the defined problem
    .run()?;
#
// print result
println!("{}", res);
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
