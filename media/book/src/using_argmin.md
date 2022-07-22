# Using argmin

In order to use argmin, one needs to add both `argmin` and `argmin-math` to `Cargo.toml`:

```toml
[dependencies]
argmin = { version = "0.6.0" }
argmin-math = { version = "0.1.0", features = ["ndarray_latest-serde,nalgebra_latest-serde"] }
```

or, for the current development version:

```toml
[dependencies]
argmin = { git = "https://github.com/argmin-rs/argmin" }
argmin-math = { git = "https://github.com/argmin-rs/argmin", features = ["ndarray_latest-serde,nalgebra_latest-serde"] }
```

Via adding `argmin-math` one can choose which math backend should be available (and whether `serde`-support is enabled or not).
The examples above use both the `ndarray_latest-serde` and `nalgebra_latest-serde` features of `argmin-math`.
For details on which options are available in `argmin-math`, please refer to the [documentation](https://docs.rs/argmin/latest/argmin-math/).


## Crate features

argmin offers a number of features which can be enabled or disabled depending on your needs.

### Default

- `slog-logger`: Support for logging observers based on [`slog`](https://crates.io/crates/slog).
- `serde1`: Support for `serde`. Needed for checkpointing and writing parameters to disk as well as logging to disk. Deactivating this feature leads to fewer dependencies and can lower compilation time, but it will also disable checkpointing and logging to disk.

### Optional 

- `ctrl`: This feature uses the `ctrlc` crate to properly stop the optimization (and return the current best result) after pressing `Ctrl+C` during an optimization run.
- `rayon`: This feature adds `rayon` as a depenceny and allows for parallel computation of cost functions, operators, gradients, Jacobians and Hessians. Note that only solvers that operate on multiple parameter vectors per iteration benefit from this feature (e.g. Particle Swarm Optimization).


### Experimental support for compiling to WebAssembly

Compiling to WASM requires the feature `wasm-bindgen`.
WASM support is still experimental. Please report any issues you encounter when using argmin in a WASM context.

## Which math backend to use

argmin offers abstractions over basic `Vec`s, `ndarray` and `nalgebra` types.
For performance reasons, the latter two should be prefered. Which one to use is a matter of taste and may depend on what you are already using. 

`Vec`s on the other hand do not have very efficient implementations for the different mathematical operations and therefore are not well suited for solvers which heavily rely on matrix operations.
However, `Vec`s are suitable for solvers such as Simulated Annealing and Particle Swarm Optimization, which mainly operate on the parameter vectors themselves. 
