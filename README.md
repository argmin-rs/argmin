[![Build Status](https://travis-ci.org/argmin-rs/argmin.svg?branch=master)](https://travis-ci.org/argmin-rs/argmin)

# argmin

This crate offers a (work in progress) numerical optimization toolbox/framework written entirely in Rust.
It is at the moment quite unstable and potentially very buggy.
Please use with care and report any bugs you encounter.
This crate is looking for contributors!
Please get in touch with me if you're interested.

[Documentation](https://argmin-rs.github.io/argmin/argmin/).


## Design goals

This crate's intention is to be useful to users as well as developers of optimization algorithms, meaning that it should be both easy to apply and easy to implement algorithms.
In particular, as a developer of optimization algorithms you should not need to worry about usability features (such as logging, dealing with different types, setters and getters for certain common parameters, counting cost function and gradient evaluations, termination, and so on).
Instead you can focus on implementing your algorithm and let `argmin-codegen` do the rest.

- Easy framework for the implementation of optimization algorithms: Define a struct to hold your data, implement a single iteration of your method and let argmin generate the rest with `#[derive(ArgminSolver)]`. This lead to similar interfaces for different solvers, making it easy for users.
- Pure Rust implementations of a wide range of optimization methods: This avoids the need to compile and interface C/C++/Fortran code.
- Type-agnostic: Many problems require data structures that go beyond simple vectors to represent the parameters. In argmin, everything is generic: All that needs to be done is implementing certain traits on your data type. For common types, these traits are already implemented.
- Convenient: Automatic and consistent logging of anything that may be important. Log to the terminal, to a file or implement your own loggers. Future plans include sending metrics to databases and connecting to big data piplines.
- Algorithm evaluation: Methods to assess the performance of an algorithm for different parameter settings, problem classes, ...

Since this crate is in a very early stage, so far most points are only partially implemented or
remain future plans.


## What is still needed?

The following list shows some of the important parts that need to be tackled in the near future:

- More optimization algorithms
- Tests
- Evaluation on real problems
- Speed improvements
- Making it ready for wasm and `#![no_std]`
- Documentation & Tutorials

Any help is appreciated! 


## Algorithms

- Line searches
  - Backtracking line search
  - More-Thuente line search
  - Hager-Zhang line search
- Trust region method
  - Cauchy point method
  - Dogleg method
  - Steihaug method
- Steepest descent
- Conjugate gradient method
- Nonlinear conjugate gradient method
- Newton methods
  - Newton's method
  - Newton-CG
- Quasi-Newton methods
  - BFGS
  - DFP
- Landweber iteration
- Simulated Annealing


## Usage

Add this to your `Cargo.toml`:

```
[dependencies]
argmin = "0.1.8"
```


### Optional features

There are additional features which can be activated in `Cargo.toml`:

```
[dependencies]
argmin = { version = "0.1.8", features = ["ctrlc", "ndarrayl"] }
```

These are currently optional, but they may move to the default features in the future. 
Without adding these features compilation to `wasm32-unknown-unkown` seems to be possible.

- `ctrlc`: Uses the `ctrlc` crate to properly stop the optimization (and return the current best result) after pressing Ctrl+C.
- `ndarrayl`: Support for `ndarray` and `ndarray-linalg`.


## License

Licensed under either of

  * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
  * MIT License ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.


### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
