+++
title = "argmin 0.8.0 and argmin-math 0.3.0 released"
description = ""
date = 2023-01-28T00:00:00+00:00
updated = 2023-01-28T00:00:00+00:00
draft = false
template = "blog/page.html"

[taxonomies]
authors = ["Stefan Kroboth"]

[extra]
+++

<b>argmin</b> is a Rust library which offers a range of numerical optimization methods and a framework for 
developing optimization algorithms. <b>argmin-math</b> is a trait-based abstraction layer for mathematical operations,
which makes argmin compatible with various math backends such as [ndarray](https://crates.io/crates/ndarray) and
[nalgebra](https://crates.io/crates/nalgebra) (or your own backend).
For details about the design of argmin and its features I suggest having a look at
[the website](https://argmin-rs.org),
[the book](https://argmin-rs.org/book),
[Github](https://github.com/argmin-rs/argmin),
[crates.io](https://crates.io/crates/argmin) and
[lib.rs](https://lib.rs/crates/argmin).

This is a short summary of the changes in argmin 0.8.0 and argmin-math 0.3.0. 
Both releases include breaking API changes; however upgrading from the previous versions should hopefully be fairly smooth.
Don't hesitate [to get in touch](https://github.com/argmin-rs/argmin/issues) in case you run into problems during upgrading.

## argmin 0.8.0

#### Improved termination handling

The solver state contains a `TerminationReason` enum which indicates why the solver terminated.
The enum variants offered in previous versions of argmin were either applicable to all solvers or were solver-specific.
[@relf](https://github.com/relf) [rightfully pointed out](https://github.com/argmin-rs/argmin/issues/305) that these variants weren't ideal:
For instance, it included an awkward variant `NotTerminated` and a couple of variants could be summarized as `Converged`.
Whenever a `Ctrl+C` was intercepted, the reason would be `Aborted`, which is suboptimal because a solver could also abort 
due to other reasons.  
In the discussion we decided to change the `TerminationReason` enum to the following:

```rust
pub enum TerminationReason {
    /// Reached maximum number of iterations
    MaxItersReached,
    /// Reached target cost function value
    TargetCostReached,
    /// Algorithm manually interrupted with Ctrl+C
    KeyboardInterrupt,
    /// Converged
    SolverConverged,
    /// Solver exit with given reason
    SolverExit(String),
}
```

The first two are potential outcomes of two checks performed by the `Executor` for every solver. 
`KeyboardInterrupt` replaces `Aborted` and is only used for `Ctrl+C`. 
`SolverConverged` indicates a successful optimization run and `SolverExit(String)` is used for 
cases where the solver stopped for a solver-specific reason specified by a string.

The awkward `NotTerminated` was dropped and instead a new enum `TerminationStatus` is introduced:

```rust
enum TerminationStatus {
    NotTerminated,
    Terminated(TerminationReason)
}
```

This enum is stored in the state instead of `TerminationReason`.
It can be obtained from an `OptimizationResult` via the solver state:

```rust
// of type &TerminationStatus
let status = result.state().get_termination_status();

// ... or ...

// of type Option<&TerminationReason>
let reason = result.state().get_termination_reason();
```

Both methods are part of the `State` trait and as such available for all available states.

Note that this is a breaking change and as such you may have to adapt your code, in particular if you use the returned 
termination reason.

Huge thanks to [@relf](https://github.com/relf) for the fruitful discussion and for doing all the heavy lifting!


#### Changes to the observer interface

In the past, values sent to the observers were only `dyn Display`, which means they could effectively only be turned into strings.
To get the actual value one had to parse the strings into a given type, which isn't great (to put it mildly).

Therefore I decided [to make the values sent to the observers typed](https://github.com/argmin-rs/argmin/pull/269) and
added support for 64bit floats, signed and unsigned 64bit integer, Booleans and Strings via the `KvValue` enum:

```rust
pub enum KvValue {
    Float(f64),
    Int(i64),
    Uint(u64),
    Bool(bool),
    Str(String),
}
```

The actual values can be retrieved via the getters `get_float`, `get_int`, `get_uint`, `get_bool` and `get_string`,
which return `Some(<inner_value>)` if the `KvValue` is of the appropriate type and `None` otherwise.

The `make_kv!` macro used in solvers to construct the Key-Value store was renamed to `kv!`.

These changes will only affect you if you wrote an observer or solver yourself.
All observers shipped with argmin should continue to work as before.

#### Other

* Implementing the `Solver` trait for a solver does not require anymore that the solver implements `serde::Serialize` when the `serde1` feature is enabled.
This was a remnant of an earlier design of the `Solver` trait found by [@relf](https://github.com/relf).
Note that checkpointing requires solvers to be serializable!
Therefore, despite lifting this requirement, it is still recommended that solvers are (de)serializable if possible.
* The check whether an optional target cost is reached is now based on the best cost function value so far rather than the current cost function value ([@relf](https://github.com/relf))
* Added a `full` feature which activates all features.
* Added a particle swarm optimization and L-BFGS example using the nalgebra backend.
* Elapsed time is now computed using `as_secs_f64` ([@TheIronBorn](https://github.com/TheIronBorn))
* Internally uses argmin-math 0.3.0, so make sure to update both!

## argmin-math 0.3.0

With this release all backends finally implement all math-related traits, meaning that every backend now works with every solver.
The only exception to this is the `vec` backend, which does not implement `ArgminInv` and as such does not
work with (Gauss-)Newton methods. 
I've spent multiple days tediously implementing all missing traits and adding tests for every implementation 
(eventually reaching 100% test coverage). 
[@hypotrochoid](https://github.com/hypotrochoid) implemented the `ArgminRandom` trait for the ndarray backend
and with that kicked off this entire endeavour. Thanks!

Apart from that support for nalgebra 0.32 was added and ndarray-linalg was updated from version 0.14 to 0.16 for 
the ndarray v0.15 backend.
Therefore you may have to update ndarray-linalg as well.

Upgrading to version 0.3.0 of argmin-math should be smooth for most cases.

## Other news

argmin is not only a collection of optimization algorithms but also aims to be a framework which facilitates 
the development of optimization algorithms.
Solvers implemented using argmin get features such as checkpointing, observers and support for various math backends for free.

However, I have not seen any use of this feature outside of argmin itself up until recently, when [@relf](https://github.com/relf)
made his [egobox-ego solver](https://crates.io/crates/egobox-ego)
[compatible with argmin](https://github.com/relf/egobox/pull/67).
This is very exciting for me as this led to valuable feedback on the design of argmin and I hope that others will follow
that example and make their solvers compatible with argmin.


<br>
<script async defer src="https://buttons.github.io/buttons.js"></script>
<p align="center">
<a class="github-button" href="https://github.com/argmin-rs/argmin" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star argmin-rs/argmin on GitHub">Star</a>
<a class="github-button" href="https://github.com/argmin-rs/argmin/subscription" data-icon="octicon-eye" data-size="large" data-show-count="true" aria-label="Watch argmin-rs/argmin on GitHub">Watch</a>
<a class="github-button" href="https://github.com/argmin-rs/argmin/fork" data-icon="octicon-repo-forked" data-size="large" data-show-count="true" aria-label="Fork argmin-rs/argmin on GitHub">Fork</a>
<a class="github-button" href="https://github.com/sponsors/stefan-k" data-icon="octicon-heart" data-size="large" aria-label="Sponsor @stefan-k on GitHub">Sponsor</a>
</p>
