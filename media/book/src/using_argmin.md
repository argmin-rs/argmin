# Using argmin

In order to use argmin, one needs to add both `argmin` and `argmin-math` to `Cargo.toml`:

```toml
[dependencies]
argmin = { version = "0.10" }
argmin-math = { version = "0.4", features = ["ndarray_latest", "nalgebra_latest"] }
```

or, for the current development version:

```toml
[dependencies]
argmin = { git = "https://github.com/argmin-rs/argmin" }
argmin-math = { git = "https://github.com/argmin-rs/argmin", features = ["ndarray_latest", "nalgebra_latest"] }
```

Via adding `argmin-math` one can choose which math backend should be available.
The examples above use both the `ndarray_latest` and `nalgebra_latest` features of `argmin-math`.
For details on which options are available in `argmin-math`, please refer to the [documentation](https://docs.rs/argmin-math/latest/argmin_math/).


## Crate features

argmin offers a number of features which can be enabled or disabled depending on your needs.


### Optional 

- `serde1`: Support for `serde`. Needed for checkpointing. Deactivating this feature leads to fewer dependencies and can lower compilation time, but it will also disable checkpointing.
- `ctrlc`: This feature uses the `ctrlc` crate to properly stop the optimization (and return the current best result) after pressing `Ctrl+C` during an optimization run.
- `rayon`: This feature adds `rayon` as a depenceny and allows for parallel computation of cost functions, operators, gradients, Jacobians and Hessians. Note that only solvers that operate on multiple parameter vectors per iteration benefit from this feature (e.g. Particle Swarm Optimization).
- `full`: Enables all default and optional features.

### Experimental support for compiling to WebAssembly

Compiling to WASM requires the feature `wasm-bindgen`.
WASM support is still experimental. Please report any issues you encounter when using argmin in a WASM context.

## Which math backend to use

argmin offers abstractions over basic `Vec`s, `ndarray` and `nalgebra` types.
For performance reasons, the latter two should be preferred. Which one to use is a matter of taste and may depend on what you are already using. 

`Vec`s on the other hand do not have very efficient implementations for the different mathematical operations and therefore are not well suited for solvers which heavily rely on matrix operations.
However, `Vec`s are suitable for solvers such as Simulated Annealing and Particle Swarm Optimization, which mainly operate on the parameter vectors themselves. 

## Examples

Every solver and most features of argmin are showcased in the [examples directory](https://github.com/argmin-rs/argmin/tree/main/examples).
Each example is a dedicated Rust crate with a corresponding `Cargo.toml`, which includes all relevant dependencies to run it.
Make sure that the examples you are looking at match the argmin version you are using be choosing the appropriate Git version tag on Github.
