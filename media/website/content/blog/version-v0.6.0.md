+++
title = "argmin 0.6.0 and argmin-math 0.1.0 released"
description = ""
date = 2022-08-01T09:19:42+00:00
updated = 2022-08-01T09:19:42+00:00
draft = false
template = "blog/page.html"

[taxonomies]
authors = ["Stefan Kroboth"]

[extra]
+++

> <b>argmin</b> offers a range of numerical optimization methods in Rust.

Due to other commitments <b>argmin</b> didn't get as much attention as it should have in previous months and years, until the end of 2021.
Having been unhappy with certain design choices for quite some time, I decided to redesign several aspects of the library.
These changes have accumulated in `argmin` version 0.6.0 and a new crate called `argmin-math`.
This blog post is an attempt to summarize what has happened; however this is by no means exhaustive. 

These changes had quite a substantial impact on the API and as such are very likely to affect you as a user.
For users of the optimization algorithms, adapting your code to the new version should be a fairly easy task with an occasional look at the documentation or the current [examples](https://github.com/argmin-rs/argmin/tree/main/argmin/examples) in the Github repository.
If you are using argmin to implement your own algorithms, the changes you will have to make are more severe. 
I suggest having a look at the source code of argmin to get a sense of how you may have to adapt your own code.

In either case, feel free to get in touch if you run into trouble, either on [Gitter](https://gitter.im/argmin-rs/community) or by opening an [issue](https://github.com/argmin-rs/argmin/issues). 


## argmin-math

One major change is that all math/algebra related code was split off into a dedicated crate called `argmin-math`.
This includes all mathematics traits as well as their implementations for various backends such as [ndarray](https://crates.io/crates/ndarray), [nalgebra](https://crates.io/crates/nalgebra) and basic `Vec`s. 
The desired backend is chosen via features (multiple versions of each backend are supported).
One just needs to add `argmin-math` with the appropriate feature to `Cargo.toml` to enable the desired backend:

```toml
[dependencies]
argmin = "0.6.0"
argmin-math = { version = "0.1.0", features = ["ndarray_v0_15-serde"] }
```

The [argmin-math documentation](https://docs.rs/argmin-math/) lists all available features.

## ArgminOp replacement

In prior versions, the `ArgminOp` trait had to be implemented for optimization problems in order to work with argmin.
It covered all (potentially) necessary operations, from computing the cost function, gradients, Jacobians and Hessians to annealing of parameter vectors in Simulated Annealing.
Users would only implement the methods which were needed by the desired solver and leave the others unimplemented, which defaulted to panicking when called.
This had a couple of disadvantages:
Firstly, it made it impossible for the solvers to communicate to the user via the trait system which methods were actually needed.
Secondly, it was limiting in the sense that the trait was not extendable by users which use argmin to implement their own solvers.

Therefore it was replaced by the dedicated traits `CostFunction`, `Gradient`, `Jacobian`, `Hessian`, `Operator`, and `Anneal`.
The solvers can now precisely define which traits must be implemented for the user-defined optimization problem.
More traits can easily be added if future solvers require them.

## Bulk computation

Cost functions, gradients and the like can now be computed in bulk for multiple input parameter vectors.
Each of the traits mentioned in the previous section provides an automatically implemented `bulk_*` method for this purpose (which can be overwritten if needed).
For instance, in case of the `CostFunction` trait, the `cost` method computes the cost function of a single parameter vector while `bulk_cost` does the same for multiple parameter vectors.
Bulk computation however must be supported by the solver, which so far is only Particle Swarm Optimization (but will also be supported by the upcoming [CMA-ES](https://github.com/argmin-rs/argmin/pull/225) solver and other future population based solvers).

By default the individual parameter vectors are processed sequentially.
Enabling the `rayon` feature enables parallel computation using [rayon](https://crates.io/crates/rayon), which can lead to a substantial performance boost depending on your problem and your hardware.
Even if `rayon` is enabled, parallel processing can selectively (for particular traits) be turned off if needed (for cases where for instance computing the `CostFunction` in parallel is desired, but not for `Gradient`).
The implementation is the result of a lot of [discussion](https://github.com/argmin-rs/argmin/issues/6) on [Github](https://github.com/argmin-rs/argmin/pull/142) which was very helpful in fleshing out the requirements of this feature.
Thanks to everyone who participated in this discussion!

## serde is now optional

Serialization and deserialization is needed for some observers and for checkpointing.
In previous versions this affected users even if no observers and no checkpointing were used.
Mainly because it required the user-defined optimization problem to implement `Serialize` and `Deserialize`, which constitutes additional effort and may not be feasible in particular cases.
Additionally, `serde` increases the compilation time.
Therefore `serde` has been made optional via the `serde1` feature which is by default enabled.
This comes with the caveat that some observers and checkpointing are not available with a disabled `serde1` feature.

## Internal state

Solvers have access to an internal state which is carried from one iteration to the next. 
This state keeps track of current and best-so-far parameter vectors, cost function values, gradients, and so on as well as function evaluation counts and the number of iterations.
In previous versions only `IterState` was available which tried to accommodate the needs of all solvers.
However, the demands of the solvers on this internal state can vary quite a lot, which lead to a somewhat awkward and potentially confusing API.

By being generic over the states, the particular kind of state can now be chosen for each solver individually.
`IterState` is still general enough to serve most solvers, but for population based methods the more appropriate `PopulationState` was introduced.
Note that the latter is so far only in use in Particle Swarm Optimization and may change substantially when other population based solvers are implemented.
More states may follow as more optimization methods are implemented.
All states must offer an interface defined by the `State` trait which covers basic functionality.
This change will typically only affect you if you use argmin to implement your own solvers.

## Initialization of solvers

The latter point also affects how solvers are initialized by a user.
While each solver is configured via its constructor and the associated methods, setting problem-related parameters such as initial parameter vectors, gradients and the like is done by initializing the internal state.
In previous versions, the `Executor` required one to provide an initial guess for the parameter vector in its constructor, which was awkward for solvers which do not require (or even accept) an initial guess.

Therefore the `Executor`s constructor was changed to only accept the solver and the optimization problem.
Configuration of the internal state is done via the `configure` method of `Executor`, where one gets access to the internal state and its methods via a closure.
The type of state offered by `configure` depends on the type of state used by the chosen solver. 

The following small example illustrates the changes.
Setting up an executor in a version prior to 0.6.0 looked like this:

```rust
let result = Executor::new(problem, solver, initial_param)
    .max_iters(1000)
    .run()?;
```

As you can see, the initial parameter vector `initial_param` was provided to the constructor and the maximum number of iterations was configured via a method of the `Executor`.
In the new version 0.6.0 this changed to:

```rust
let result = Executor::new(problem, solver)
    .configure(|state| state.param(initial_param).max_iters(1000))
    .run()?;
```

Both initial parameter vector and the maximum number of iterations are directly provided to the `state` using `Executor`s `configure` method.

Note that depending on the solver, more aspects can be initialized. For instance, one can also provide an initial gradient which skips the first computation of the gradient.

## Checkpointing 

Checkpointing was redesigned from the ground up and is now substiantially easier to configure: 

```rust
let result = Executor::new(problem, solver)
    .configure(|state| state.param(initial_param).max_iters(1000))
    .checkpointing(checkpoint)
    .run()?;
```

The variable `checkpoint` represents an instance of something that implements the `Checkpoint` trait.
argmin itself offers `FileCheckpoint` which saves the checkpoint to disk.
Thanks to the `Checkpoint` trait you can also implement your own checkpointing mechanisms.

## Problem replaces OpWrapper

`OpWrapper` was used internally by the `Executor` to wrap the user-defined optimization problem and count the number of function evaluations. 
It was renamed to `Problem` and comes with a more consistent interface.
`Problem` itself implements `CostFunction`, `Gradient`, `Jacobian`, `Hessian`, `Operator`, and `Anneal` and can thus be used like the user-defined problem itself.
`Problem` still counts the function evaluations (also for bulk evaluation). 

This change is unlikely to affect you, unless you use argmin to implement your own algorithms.


## Brent's optimization method

[@Armavica](https://github.com/Armavica) added Brent's optimization method (Thanks!).
In the course of this the previously implemented Brent's method (the root finding method) was renamed to `BrentRoot`.


## Documentation

Adding missing documentation after years of development really teaches you a lesson.
I've spent countless hours adding (hopefully useful) documentation and doctests to every struct, method, function, module and trait. 
The documentation on [docs.rs](https://docs.rs/argmin) now covers the API and the newly created [argmin book](https://www.argmin-rs.org/book/) is a set of loosely coupled tutorials on how to use different aspects of the library.
I plan to continuously improve and add to the book such that it eventually becomes a valuable resource for anyone using argmin.

While the situation has improved a lot in terms of documentation, there are still corners which could use a bit more love.
Feedback by users is essential for improving the quality of both the book and the documentation.
Therefore please get in touch if you have suggestions or if you even want to contribute to the documentation.
This is highly appreciated.

I've also updated the website, which still isn't great design-wise, but time is limited and there are more enjoyable things to do ;-)



## Other changes

* Removed the `prelude` module. Alternatively `argmin::core::*` can be used, although it is not recommended.
* Some interfaces were redesigned to avoid cloning.
* `ArgminResult` was renamed to `OptimizationResult` and contains the solver, the problem and the final state of the optimization run.
* The `Solver` trait was redesigned to accommodate for the new handling of (generic) internal state.
* `KV` does not store `String`s anymore, but instead elements which are `dyn Display`. Conversion into strings is done by the observer, which has the advantage that there are no computational costs for conversion when no observer is used. This will likely be further improved in the future, where the `KV` will be able to store any type.
* Improved the API surface to be more consistent in terms of naming.
* Particle Swarm Optimization was redesigned and uses `PopulationState` internally. Also, it is now able to compute the cost function for all individuals in the population in parallel using the above mentioned bulk processing methods.
* Some of the solvers had unnecessary trait bounds which were removed.
* Removed many `unwrap()`s and replaced them with meaningful errors.
* The optional `stdweb` dependency was removed.
* Many more tests and doc tests were added.
* `ArgminError` is now `non_exhaustive`.
* nalgebra 0.31 is supported. 
* Many now useless macros were removed.
* Improved CI.
* Added a Code of Conduct (finally).
* Created a new logo which only really works in dark mode. Who still uses light mode anyways? ;-)
* Fixed a bug in Nelder-Mead reported by [@bivhitscar](https://github.com/bivhitscar) (Thanks!).
* Unnecessary `Default` trait bound on `TrustRegion` and `Steihaug` was removed by [@cfunky](https://github.com/cfunky) (Thanks!).
* All argmin crates now follow SemVer strictly (I have to admit that it needed [complaints](https://github.com/argmin-rs/argmin/issues/135) to fully commit to SemVer).


## Conclusion

Version 0.6.0 is a huge release for me personally because I got to make some substantial improvements to the library which hopefully made the interface more consistent and the entire code base easier to maintain.
My hope is that the library is now better approachable by both users and potential contributors.
Some design aspects which need to be addressed remain, and new problems will certainly arise, but for now I hope to be able to spend my time on more fun aspects of this project.

#### What to expect next
In the future you can expect to see more regular releases, with fewer severe changes. 

One of the next steps will be a redesign of how stopping criteria work.
The goal is to enable users to supply their own stopping criteria.

I look forward to implementing more solvers, in particular for constrained optimization which is not yet possible (apart from PSO and SA). 

An implementation of [CMA-ES](https://github.com/argmin-rs/argmin/pull/225) by [@VolodymyrOrlov](https://github.com/VolodymyrOrlov) is almost ready to be merged.

I hope that the changes and the improved documentation make it easier for users to give feedback and for potential contributors to join the project.
Please don't hesitate to get in touch if you are looking for help, have feedback or want to contribute.

I also plan to share potential projects on the issue tracker and label them as `good-first-issue` if appropriate.
I will offer mentoring for newcomers if my schedule allows it.

Thanks again to everyone who contributed by issuing PRs, giving feedback and discussing issues!
I really appreciate it.


<br>
<script async defer src="https://buttons.github.io/buttons.js"></script>
<p align="center">
<a class="github-button" href="https://github.com/argmin-rs/argmin" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star argmin-rs/argmin on GitHub">Star</a>
<a class="github-button" href="https://github.com/argmin-rs/argmin/subscription" data-icon="octicon-eye" data-size="large" data-show-count="true" aria-label="Watch argmin-rs/argmin on GitHub">Watch</a>
<a class="github-button" href="https://github.com/argmin-rs/argmin/fork" data-icon="octicon-repo-forked" data-size="large" data-show-count="true" aria-label="Fork argmin-rs/argmin on GitHub">Fork</a>
<a class="github-button" href="https://github.com/sponsors/stefan-k" data-icon="octicon-heart" data-size="large" aria-label="Sponsor @stefan-k on GitHub">Sponsor</a>
</p>
