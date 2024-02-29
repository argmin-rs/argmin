<p align="center">
  <img
    width="400"
    src="https://raw.githubusercontent.com/argmin-rs/argmin/main/media/logo.png"
  />
</p>
<p align="center">
    Mathematical optimization in pure Rust
</p>

<p align="center">
  <a href="https://argmin-rs.org">Website</a>
  |
  <a href="https://argmin-rs.org/book/">Book</a>
  |
  <a href="https://docs.rs/argmin">Docs (latest release)</a>
  |
  <a href="https://argmin-rs.github.io/argmin/argmin/">Docs (main branch)</a>
  |
  <a href="https://github.com/argmin-rs/argmin/tree/argmin-v0.10.0/examples">Examples (latest release)</a>
  |
  <a href="https://github.com/argmin-rs/argmin/tree/main/examples">Examples (main branch)</a>
</p>

<p align="center">
<!--
  <a href="https://argmin-rs.org"
    ><img
      src="https://img.shields.io/website?down_message=offline&style=flat-square&up_message=argmin-rs.org&url=http%3A%2F%2Fargmin-rs.org"
      alt="Website"
  /></a>
  <a href="https://argmin-rs.org/book/"
    ><img
      src="https://img.shields.io/website?label=book&style=flat-square&url=http%3A%2F%2Fargmin-rs.org%2Fbook%2F"
      alt="Website"
  /></a>
--!>
  <a href="https://crates.io/crates/argmin"
    ><img
      src="https://img.shields.io/crates/v/argmin?style=flat-square"
      alt="Crates.io version"
  /></a>
<!--
  <a href="https://docs.rs/argmin"
    ><img
      src="https://img.shields.io/docsrs/argmin?style=flat-square&label=docs.rs"
      alt="Documentation of latest release"
  /></a>
  <a href="https://argmin-rs.github.io/argmin/argmin/"
    ><img
      src="https://img.shields.io/docsrs/argmin?style=flat-square&label=docs main branch"
      alt="Documentation of main branch"
  /></a>
--!>
  <a href="https://crates.io/crates/argmin"
    ><img
      src="https://img.shields.io/crates/d/argmin?style=flat-square"
      alt="Crates.io downloads"
  /></a>
  <a href="https://github.com/argmin-rs/argmin/actions"
    ><img
      src="https://img.shields.io/github/actions/workflow/status/argmin-rs/argmin/ci.yml?branch=main&label=argmin CI&style=flat-square"
      alt="GitHub Actions workflow status"
  /></a>
  <img
    src="https://img.shields.io/crates/l/argmin?style=flat-square"
    alt="License"
  />
  <a href="https://discord.gg/fYB8AwxxMW"
    ><img
      src="https://img.shields.io/discord/1189119565335109683?style=flat-square&label=argmin%20Discord"
      alt="argmin Discord"
  /></a>
</p>


argmin is a numerical optimization library written entirely in Rust.

argmins goal is to offer a wide range of optimization algorithms with a consistent interface.
It is type-agnostic by design, meaning that any type and/or math backend, such as `nalgebra` or `ndarray` can be used -- even your own.

Observers allow one to track the progress of iterations, either by using one of the provided ones for logging to screen or disk or by implementing your own.

An optional checkpointing mechanism helps to mitigate the negative effects of crashes in unstable computing environments.

Due to Rusts powerful generics and traits, most features can be exchanged by your own tailored implementations.

argmin is designed to simplify the implementation of optimization algorithms and as such can also be used as a toolbox for the development of new algorithms. One can focus on the algorithm itself, while the handling of termination, parameter vectors, populations, gradients, Jacobians and Hessians is taken care of by the library.


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
  - Newton’s method
  - Newton-CG
- Quasi-Newton methods
  - BFGS
  - L-BFGS
  - DFP
  - SR1
  - SR1-TrustRegion
- Gauss-Newton method
- Gauss-Newton method with linesearch
- Golden-section search
- Landweber iteration
- Brent’s method
- Nelder-Mead method
- Simulated Annealing
- Particle Swarm Optimization

### External solvers compatible with argmin

External solvers which implement the `Solver` trait are compatible with argmins `Executor`, 
and as such can leverage features like checkpointing and observers. 

- [egobox](https://crates.io/crates/egobox-ego)
- [cobyla](https://crates.io/crates/cobyla)


## License

Licensed under either of

 - Apache License, Version 2.0, ([LICENSE-APACHE](https://github.com/argmin-rs/argmin/blob/main/LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
 - MIT License ([LICENSE-MIT](https://github.com/argmin-rs/argmin/blob/main/LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.


### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
