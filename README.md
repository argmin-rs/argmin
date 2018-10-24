[![Build Status](https://travis-ci.org/argmin-rs/argmin.svg?branch=master)](https://travis-ci.org/argmin-rs/argmin)

# argmin

A work-in-progress optimization toolbox written in Rust.
Highly unstable and potentially *very buggy*.
Please use with care and report any bugs you encounter.
This crate is looking for contributors!
Please get in touch with me if you're interested.

[Documentation](https://argmin-rs.github.io/argmin/argmin/).


## Design goals

This crate's intention is to be useful to users as well as developers of optimization algorithms, meaning that it should be both easy to apply and easy to implement algoritms.
In particular, as a developer of optimization algorithms you should not need to worry about usability features (such as logging, dealing with different types, setters and getters for certain common parameters, counting cost function and gradient evaluations, termination,  and so on).
Instead you can focus on implementing your algorithm and let argmin do the boring stuff for you.

- Provide an easy framework for the implementation of optimization algorithms: Define a struct to hold your data, implement a single iteration of your method and let argmin generate the rest with `#[derive(ArgminSolver)]`. With is approach, the interfaces to different solvers will be fairly similar, making it easy for users to try different methods on their problem without much work.
- Provide pure Rust implementations of many optimization methods. That way there is no need to compile and interface C code and it furthermore avoids inconsistent interfaces.
- Be type-agnostic: If you have your own special type that you need for solving your optimization problem, you just need to implement a couple of traits on that type and you're ready to go. These traits will already be implemented for common types.
- Easy iteration information logging: Either print your iteration information to the terminal, or write it to a file, or store it in a database or send it to a big data pipeline.
- Easy evaluation of algorithms: Make it possible to run algorithms with different parameters and store all necessary of information of all iterations and calculate measures in order to evaluate the performance of the implementation/method. Take particular care of stochastic methods.

Since this crate is in a very early stage, so far most points are only partially implemented. In addition it is very likely *very buggy*.


## What is still needed?

The following list shows some of the important parts that need to be tackled in the near future:

- (Much, much) more optimization algorithms
- Tests, tests, tests!
- Evaluation on real problems
- Strong focus on speed
- Making it ready for wasm and `#![no_std]`
- Useful documentation
- Extensive tutorials

Any help is appreciated! 


## Algorithms

- [X] Linesearches
  - [X] Backtracking line search
  - [X] More-Thuente line search
  - [X] Hager-Zhang line search
- [X] Trust region method
  - [X] Cauchy point method
  - [X] Dogleg method
  - [X] Steihaug method
- [X] Steepest Descent
- [X] Conjugate Gradient method
- [X] Simulated Annealing


## License

Licensed under either of

  * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
  * MIT License ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
