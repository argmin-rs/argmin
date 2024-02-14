# Running a solver

The [`Executor`](https://docs.rs/argmin/latest/argmin/core/struct.Executor.html)s constructor takes a solver and an optimization problem as input and applies the solver to the problem.
The initial state of the optimization run can be modified via the [`configure`](https://docs.rs/argmin/latest/argmin/core/struct.Executor.html#method.configure) method.
This method accepts a closure with the state as only parameter.
This allows one to provide initial parameter vectors, cost function values, Hessians, and so on via the closure.
There are different kinds/types of state and the particular kind of state used depends on the solver.
Most solvers internally use [`IterState`](https://docs.rs/argmin/latest/argmin/core/struct.IterState.html), but some (for instance Particle Swarm Optimization) use [`PopulationState`](https://docs.rs/argmin/latest/argmin/core/struct.PopulationState.html).
Please refer to the respective documentation for details on how to modify the state.

Once the `Executor` is configured, the optimization is run via the [`run`](https://docs.rs/argmin/latest/argmin/core/struct.Executor.html#method.run) method.
This method returns an [`OptimizationResult`](https://docs.rs/argmin/latest/argmin/core/struct.OptimizationResult.html) which contains the provided problem, the solver and the final state. 
Assuming the variable is called `res`, the final parameter vector can be accessed via `res.state().get_best_param()` and the corresponding cost function value via `res.state().get_best_cost()`.

For an overview, `OptimizationResult`s `Display` implementation can be used to print the result: `println!("{}", res)`.

The following example shows how to use the `SteepestDescent` solver to solve a problem which implements `CostFunction` and `Gradient` (which are both required by the solver).

```rust
# #![allow(unused_imports)]
# extern crate argmin;
# extern crate argmin_testfunctions;
use argmin::core::{State, Error, Executor, CostFunction, Gradient};
use argmin::solver::gradientdescent::SteepestDescent;
use argmin::solver::linesearch::MoreThuenteLineSearch;
# use argmin_testfunctions::{rosenbrock, rosenbrock_derivative};

struct MyProblem {}

// Implement `CostFunction` for `MyProblem`
impl CostFunction for MyProblem {
      // [...]
#     /// Type of the parameter vector
#     type Param = Vec<f64>;
#     /// Type of the return value computed by the cost function
#     type Output = f64;
#
#     /// Apply the cost function to a parameter `p`
#     fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
#         Ok(rosenbrock(p))
#     }
}

// Implement `Gradient` for `MyProblem`
impl Gradient for MyProblem {
      // [...]
#     /// Type of the parameter vector
#     type Param = Vec<f64>;
#     /// Type of the return value computed by the cost function
#     type Gradient = Vec<f64>;
#
#     /// Compute the gradient at parameter `p`.
#     fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
#         Ok(rosenbrock_derivative(p).to_vec())
#     }
}
#
# fn run() -> Result<(), Error> {

// Create new instance of cost function
let cost = MyProblem {};
 
// Define initial parameter vector
let init_param: Vec<f64> = vec![-1.2, 1.0];
 
// Set up line search needed by `SteepestDescent`
let linesearch = MoreThuenteLineSearch::new();
 
// Set up solver -- `SteepestDescent` requires a linesearch
let solver = SteepestDescent::new(linesearch);
 
// Create an `Executor` object 
let res = Executor::new(cost, solver)
    // Via `configure`, one has access to the internally used state.
    // This state can be initialized, for instance by providing an
    // initial parameter vector.
    // The maximum number of iterations is also set via this method.
    // In this particular case, the state exposed is of type `IterState`.
    // The documentation of `IterState` shows how this struct can be
    // manipulated.
    // Population based solvers use `PopulationState` instead of 
    // `IterState`.
    .configure(|state|
        state
            // Set initial parameters (depending on the solver,
            // this may be required)
            .param(init_param)
            // Set maximum iterations to 10
            // (optional, set to `std::u64::MAX` if not provided)
            .max_iters(10)
            // Set target cost. The solver stops when this cost
            // function value is reached (optional)
            .target_cost(0.0)
    )
    // run the solver on the defined problem
    .run()?;

// print result
println!("{}", res);

// Extract results from state

// Best parameter vector
let best = res.state().get_best_param().unwrap();

// Cost function value associated with best parameter vector
let best_cost = res.state().get_best_cost();

// Check the execution status
let termination_status = res.state().get_termination_status();

// Optionally, check why the optimizer terminated (if status is terminated) 
let termination_reason = res.state().get_termination_reason();

// Time needed for optimization
let time_needed = res.state().get_time().unwrap();

// Total number of iterations needed
let num_iterations = res.state().get_iter();

// Iteration number where the last best parameter vector was found
let num_iterations_best = res.state().get_last_best_iter();

// Number of evaluation counts per method (Cost, Gradient)
let function_evaluation_counts = res.state().get_func_counts();
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

Optionally, `Executor` allows one to terminate a run after a given timeout, which can be set with the `timeout` method of `Executor`. 
The check whether the overall runtime exceeds the timeout is performed after every iteration, therefore the actual runtime can be longer than the set timeout.
In case of timeout, the run terminates with `TerminationReason::Timeout`.
