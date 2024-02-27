# Checkpointing

Checkpointing is a useful mechanism for mitigating the effects of crashes when software is run in an unstable environment, particularly for long run times.
Checkpoints are snapshots of the current state of the optimization which can be resumed from in case of a crash.
These checkpoints are saved regularly at a user-chosen frequency.

Currently only saving checkpoints to disk with [`FileCheckpoint`](https://docs.rs/argmin-checkpointing-file/latest/argmin_checkpointing_file/struct.FileCheckpoint.html) is provided
in the [`argmin-checkpointing-file`](https://crates.io/crates/argmin-checkpointing-file) crate.
Via the [`Checkpoint`](https://docs.rs/argmin/latest/argmin/core/checkpointing/trait.Checkpoint.html) trait other checkpointing approaches can be implemented (see the chapter on [implementing a checkpointing method](./implementing_checkpointing.md) for details).

The [`CheckpointingFrequency`](https://docs.rs/argmin/latest/argmin/core/checkpointing/enum.CheckpointingFrequency.html) defines how often checkpoints are saved and can be chosen to be either `Always` (every iteration), `Every(u64)` (every Nth iteration) or `Never`.

The following example shows how the [`checkpointing`](https://docs.rs/argmin/latest/argmin/core/struct.Executor.html#method.checkpointing) method is used to configure and activate checkpointing.
If no checkpoint is available on disk yet, an optimization will be started from scratch.
If the run crashes and a checkpoint is found on disk, then it will resume from the checkpoint.

## Example

```rust
# extern crate argmin;
# extern crate argmin_testfunctions;
# use argmin::core::{CostFunction, Error, Executor, Gradient, observers::ObserverMode};
# #[cfg(feature = "serde1")]
use argmin::core::checkpointing::CheckpointingFrequency;
use argmin_checkpointing_file::FileCheckpoint;
# use argmin_observer_slog::SlogLogger;
# use argmin::solver::landweber::Landweber;
# use argmin_testfunctions::{rosenbrock, rosenbrock_derivative};
#
# #[derive(Default)]
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
#         Ok(rosenbrock_derivative(p).to_vec())
#     }
# }
#
# fn run() -> Result<(), Error> {
#     // define initial parameter vector
#     let init_param: Vec<f64> = vec![1.2, 1.2];
#     let my_optimization_problem = Rosenbrock {};
#
#     let iters = 35;
#     let solver = Landweber::new(0.001);
// [...]

# #[cfg(feature = "serde1")]
let checkpoint = FileCheckpoint::new(
    // Directory
    ".checkpoints",
    // File base name
    "optim",
    // How often to save a checkpoint (in this case every 20 iterations)
    CheckpointingFrequency::Every(20)
);

#
# #[cfg(feature = "serde1")]
let res = Executor::new(my_optimization_problem, solver)
    .configure(|state| state.param(init_param).max_iters(iters))
    .checkpointing(checkpoint)
    .run()?;

// [...]
#
#     Ok(())
# }
#
# fn main() {
#     if let Err(ref e) = run() {
#         println!("{}", e);
#     }
# }
```
