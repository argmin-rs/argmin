+++
title = "argmin 0.10.0 and argmin-math v0.4.0"
description = ""
date = 2024-02-27T00:00:00+00:00
updated = 2024-02-27T00:00:00+00:00
draft = false
template = "blog/page.html"

[taxonomies]
authors = ["Stefan Kroboth"]

[extra]
+++

<b>argmin</b> is a Rust library which offers a range of numerical optimization methods and is a framework for 
developing optimization algorithms.  Details about the design and features of argmin can be found on
[the website](https://argmin-rs.org),
in the [book](https://argmin-rs.org/book),
on [Github](https://github.com/argmin-rs/argmin),
on [crates.io](https://crates.io/crates/argmin) and
on [lib.rs](https://lib.rs/crates/argmin).

## argmin 0.10.0

argmin has seen a few smaller, but partially breaking changes in this release.

##### Observers and Checkpointing (Breaking)

Observers and checkpointing were moved from argmin into dedicated crates. This greatly reduced the
complexity of argmin and the number of its dependencies. Now users can decide which observers
and checkpointing they actually need, thus potentially reducing compile times and code size.

Notable changes:
* `SlogLogger` is now part of the [argmin-observer-slog](https://crates.io/crates/argmin-observer-slog) crate.
* `WriteToFile` was renamed to `ParamWriter` and is now part of the
[argmin-observer-paramwriter](https://crates.io/crates/argmin-observer-paramwriter) crate.
* `FileCheckpoint` is now part of the
[argmin-checkpointing-file](https://crates.io/crates/argmin-checkpointing-file) crate.

Furthermore, `Observe::observe_init`, which is called after initialization of the solver, now also
has access to the `state`, meaning that it can observe the initial state of the optimization run.

##### Interrupt handling (Breaking)

Interrupt handling now includes `SIGINT`, `SIGTERM` and `SIGHUP`. Consequently,
`TerminationReason::KeyboardInterrupt` was renamed to `TerminationReason::Interrupt`.
This is a breaking change for those who match on `TerminationReason`.

##### Optional timeout

`Executor` now allows one to terminate a run after a given timeout, which can be set with the `timeout` method of `Executor`. 
The check whether the overall runtime exceeds the timeout is performed after every iteration,
therefore the actual runtime can be longer than the set timeout.
In case of timeout, the run terminates with `TerminationReason::Timeout`.

The timeout is set via the `timeout` method of `Executor`:

```rust
let res = Executor::new(operator, solver)
    .timeout(std::time::Duration::from_secs(3))
    .run()?;
```

##### The optional `serde1` feature

With moving observers out of the argmin crate, it became easier to reason about the optional
`serde1` feature, which eventually led to the removal of the `SerializeAlias` and `DeserializeAlias` traits.
Activating `serde1` for argmin is now only necessary for checkpointing.

##### Other changes and fixes

* In `GaussNewton`, the residuals were out of sync with the parameter vector
(Thanks [@gmilleramilar](https://github.com/gmilleramilar) for reporting
and starting the fix).

* A failing line search in `LBFGS` now does not lead to the error propagating to the 
`Executor` but instead causes `LBFGS` to terminate with `TerminationReason::SolverExit(string)` where
`string` indicates the reason why the line search failed.

* The random number generator (RNG) used in `ParticleSwarm` can now be set by 
users (Thanks to [@jonboh](https://github.com/jonboh)).

* All crates are now in the `crates` directory of the argmin monorepo.

* All examples are now in dedicated crates in the `examples` directory.
This makes it easier for users to see which dependencies (and features) an example requires.

* argmin-math dependency updated to version 0.4.

## argmin-math 0.4.0

##### Removal of `*-serde` features (Breaking)

All features ending with `-serde` were removed. Support for serde is now enabled simply be enabling
serde support in the backends (`ndarray` or `nalgebra`). Using `*-serde` features will cause a failing
build, which can be solved without loss of functionality by removing `-serde` from the feature.

##### Development on Windows

Thanks to [@Tastaturtaste](https://github.com/Tastaturtaste) development on Windows is finally possible.
This was achieved by using MKL in the `argmin-math` tests and involved a lot of effort to be able to continue
supporting multiple versions of ndarray.

##### Other changes (Breaking)

* `ArgminInv` is now also implemented for scalars (`f32` and `f64`, thanks to [@sdrap](https://github.com/sdrap)).
* `ArgminRandom::rand_from_range(...)` now also accepts a random number generator. This allows for setting 
the seed manually (Thanks to [@jonboh](https://github.com/jonboh)).
* Tests for `ArgminMinMax` were added (Thanks to [@Shreyan11](https://github.com/Shreyan11)).

## Contributing

There currently are [a couple of open issues](https://github.com/argmin-rs/argmin/issues), some of which
are good first issues to start getting involved in argmin. I am happy to provide guidance if needed.

Suggestions and feature requests are also welcome. For instance, let me know which observers
or checkpointing methods yo would like to see.

## Conclusion

This release mostly affected the project structure, which was an important step for further development.
By moving observers and checkpointing into dedicated crates, the dependencies of argmin could be reduced 
and users have more freedom in pulling in only what they need.
The changes to `ParticleSwarm` and `ArgminRandom` allow for reproducible runs by setting an RNG seed.
Interrupt handling has become more useful by capturing more interrupts.
Finally, development on Windows is possible.

This release included contributions from
[@gmilleramilar](https://github.com/gmilleramilar),
[@jonboh](https://github.com/jonboh),
[@Tastaturtaste](https://github.com/Tastaturtaste),
[@Shreyan11](https://github.com/Shreyan11),
[@sdrap](https://github.com/sdrap), and
[@stefan-k](https://github.com/stefan-k).
Thanks to the contributors and those who opened and responded to issues and discussions!

A bit offtopic, but I want to use this opportunity to give a shoutout to
[gkls-rs](https://crates.io/crates/gkls-rs), a pure Rust implementation of the GKLS function
generator by [@jonboh](https://github.com/jonboh).

## Discord server

If you're interested you're invited to join the  [Discord](https://discord.gg/fYB8AwxxMW)!


<br>
<script async defer src="https://buttons.github.io/buttons.js"></script>
<p align="center">
<a class="github-button" href="https://github.com/argmin-rs/argmin" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star argmin-rs/argmin on GitHub">Star</a>
<a class="github-button" href="https://github.com/argmin-rs/argmin/subscription" data-icon="octicon-eye" data-size="large" data-show-count="true" aria-label="Watch argmin-rs/argmin on GitHub">Watch</a>
<a class="github-button" href="https://github.com/argmin-rs/argmin/fork" data-icon="octicon-repo-forked" data-size="large" data-show-count="true" aria-label="Fork argmin-rs/argmin on GitHub">Fork</a>
<a class="github-button" href="https://github.com/sponsors/stefan-k" data-icon="octicon-heart" data-size="large" aria-label="Sponsor @stefan-k on GitHub">Sponsor</a>
</p>
