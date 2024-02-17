# Introduction


argmin is a numerical optimization library written entirely in Rust.

Its goal is to offer a wide range of optimization algorithms with a consistent interface. 
It is type-agnostic by design, meaning that any type and/or math backend, such as `nalgebra` or `ndarray` can be used -- even your own.

Observers allow one to track the progress of iterations, either by using one of the provided ones for logging to screen or disk or by implementing your own.

An optional checkpointing mechanism helps to mitigate the negative effects of crashes in unstable computing environments.

Due to Rusts powerful generics and traits, most features can be exchanged by your own tailored implementations.

argmin is designed to simplify the implementation of optimization algorithms and as such can also be used as a toolbox for the development of new algorithms. One can focus on the algorithm itself, while the handling of termination, parameter vectors, populations, gradients, Jacobians and Hessians is taken care of by the library.

> **IMPORTANT NOTE**
>
> This book covers version 0.10 of argmin! Parts of this book may not apply to versions below 0.10.


## The argmin ecosystem

The ecosystem consists of a number of crates:

* [argmin](https://crates.io/crates/argmin): Optimization algorithms and framework
* [argmin-math](https://crates.io/crates/argmin-math): Interface for math backend abstraction and implementations for various versions of [ndarray](https://crates.io/crates/ndarray), [nalgebra](https://crates.io/crates/nalgebra) and `Vec`s.
* [argmin-testfunctions](https://crates.io/crates/argmin-testfunctions): A collection of test functions
* [finitediff](https://crates.io/crates/finitediff): Finite differentiation
* [modcholesky](https://crates.io/crates/modcholesky): Modified cholesky decompositions


## Algorithms

argmin comes with a number of line searches (Backtracking, More-Thuente, Hager-Zhang, trust region methods (Cauchy point, Dogleg, Steihaug), Steepest descent, (Nonlinear) conjugate gradient, Newton method, Newton-CG, Quasi-Newton methods (BFGS, L-BFGS, DFP, SR1-TrustRegion), Gauss-Newton methods (with and without line search), Golden-section search, Landweber, Brents optimization and root finding methods, Nelder-Mead, Simulated Annealing, Particle Swarm Optimization and CMA-ES.

For a complete and up-to-date list of all algorithms please visit the [API documentation](https://docs.rs/argmin/latest/argmin/).

> Examples for each algorithm can be found on [Github](https://github.com/argmin-rs/argmin/tree/main/examples). Make sure to choose the tag matching the argmin version you are using.

## Documentation

This book is a guide on how to use argmins algorithms as well as on how to implement algorithms using argmins framework. 
For detailed information on specific algorithms or traits, please refer to [argmins API documentation](https://docs.rs/argmin/latest/argmin/). 

The [argmin-math documentation](https://docs.rs/argmin/latest/argmin-math/) outlines the abstractions over the math backends and the [argmin-testfunctions API documentation](https://docs.rs/argmin/latest/argmin-testfunctions/) lists all available test functions.

For details on how to use `finitediff` for finite differentiation, please refer to the corresponding [API documentation](https://docs.rs/argmin/latest/finitediff/).

The documentation of `modcholesky` can be found [here](https://docs.rs/argmin/latest/modcholesky).

## Discord server

Feel free to join the [argmin Discord](https://discord.gg/fYB8AwxxMW)!

## License

Both the source code as well as the documentation are licensed under either of

 - Apache License, Version 2.0, ([LICENSE-APACHE][__link35] or <http://www.apache.org/licenses/LICENSE-2.0>)
 - MIT License ([LICENSE-MIT][__link37] or <http://opensource.org/licenses/MIT>)

at your option.

#### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
